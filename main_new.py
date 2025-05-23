# main_refactored.py

import copy
import json
import random
import logging
from concurrent.futures import ThreadPoolExecutor

# Assuming r1.py is in the same directory or accessible in PYTHONPATH
from r1 import get_openai_api_response_cot

from ours.from_yuxian_chinese_openrouter_new_prompts.config import (
    axis as ALL_CHALLENGE_AXES,
    Inference_Memory_Topic,
    # Assuming these are also defined for completeness
    Inference_Memory_Eval_Config,
    # Assuming these
    PERSONL_SEEDS_LIST
)
from planer import prompt as PLANNER_BASE_PROMPT
from user import prompt as USER_BASE_PROMPT

# Setup logging for better traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_CONVERSATION_HISTORY_TURNS_FOR_PLANNER = 10 # To prevent planner prompt from getting too long 这个常量用于限制在更新Planner Agent的蓝图时，传递给Planner Agent的对话历史的长度（轮数）。这是为了防止传递给Planner Agent的Prompt过长，从而超出LLM的上下文窗口限制或导致处理效率低下。一轮包含用户和助手的发言，所以* 2。

def format_planner_prompt(base_prompt_template, challenge_axis, topic_details,
                          evaluation_config, max_dialogue_turns, persona_seed):
    """Formats the planner agent's prompt."""
    prompt = base_prompt_template.replace("{axis}", challenge_axis)
    prompt = prompt.replace("{topic}", json.dumps(topic_details, indent=4, ensure_ascii=False))
    prompt = prompt.replace("{num_turns}", str(max_dialogue_turns))
    prompt = prompt.replace("{Definition}", evaluation_config["category_definition"])
    prompt = prompt.replace("{Pass Criteria}", evaluation_config["pass_criteria"])
    prompt = prompt.replace("{Fail Criteria}", evaluation_config["failure_criteria"])
    prompt = prompt.replace("{Example}", evaluation_config["example"])
    prompt = prompt.replace("{Persona_seed}", persona_seed)
    return prompt

def format_user_agent_prompt(base_prompt_template, challenge_axis, current_blueprint,
                             topic_details, max_dialogue_turns):
    """Formats the user agent's prompt."""
    prompt = base_prompt_template.replace("{axis}", challenge_axis) # Assuming user prompt also takes a single axis string
    prompt = prompt.replace("{topic}", json.dumps(topic_details, indent=4, ensure_ascii=False))
    prompt = prompt.replace("{num_turns}", str(max_dialogue_turns)) # Though num_turns might be more for planner
    prompt = prompt.replace("{blueprint}", current_blueprint)
    return prompt

def parse_user_agent_response(response_text):
    """Parses the User Agent's structured response to extract message and stop flag."""
    import re
    user_message = None
    stop_flag = False # Default to False

    # Regex to find User Message, allowing for variations in quotes or no quotes
    # Making it more robust: looking for "User Message:" possibly followed by ** then optional quote then message then optional quote
    msg_match = re.search(r'User Message:\*?\*?\s*"?([^"\n]*?)"?\s*(\n|$)', response_text, re.DOTALL | re.IGNORECASE)
    if not msg_match: # Try another common pattern if the first fails
        msg_match = re.search(r'User Message:\s*(.*?)(\n•\s*STOP:|$)', response_text, re.DOTALL | re.IGNORECASE)

    if msg_match:
        user_message = msg_match.group(1).strip()
        logging.info(f"Extracted User Message: {user_message}")
    else:
        logging.warning(f"Could not extract User Message from: {response_text}")
        # Fallback: consider the whole text as message if specific parsing fails, or handle error
        # For now, let's assume if no "User Message:" is found, it's an issue.
        user_message = "Error: Could not parse user message."


    stop_match = re.search(r'STOP:\*?\*?\s*(True|False)', response_text, re.IGNORECASE)
    if stop_match:
        stop_flag_str = stop_match.group(1).strip().lower()
        stop_flag = stop_flag_str == 'true'
        logging.info(f"Extracted STOP flag: {stop_flag}")
    else:
        logging.warning(f"Could not extract STOP flag from: {response_text}, defaulting to False.")

    return user_message, stop_flag

def build_conversation_history_text(direct_dialogue_turns):
    """Builds a text representation of the conversation history for the planner."""
    history_text = "\nHere is the conversation history up to the last assistant response:\n"
    # Limit the history sent to planner to avoid excessive length
    start_index = max(0, len(direct_dialogue_turns) - MAX_CONVERSATION_HISTORY_TURNS_FOR_PLANNER * 2) # Each turn is user + assistant
    for i in range(start_index, len(direct_dialogue_turns)):
        turn = direct_dialogue_turns[i]
        history_text += f"{turn['role'].capitalize()}: {turn['content']}\n"
    return history_text.strip()

def build_planner_update_messages(initial_planner_prompt_content, previous_planner_blueprint,
                                  current_direct_dialogue_turns):
    """Constructs the message list for updating the planner's blueprint."""
    messages = [{"role": "user", "content": initial_planner_prompt_content}]
    if previous_planner_blueprint: # Add previous blueprint as assistant's last thought
        messages.append({"role": "assistant", "content": previous_planner_blueprint})
    messages.append({"role": "user", "content": build_conversation_history_text(current_direct_dialogue_turns)})
    return messages

def build_user_agent_update_messages(initial_user_agent_prompt_content, previous_user_agent_full_response,
                                     updated_planner_blueprint):
    """Constructs the message list for the user agent to generate the next turn."""
    messages = [{"role": "user", "content": initial_user_agent_prompt_content}]
    if previous_user_agent_full_response: # Add previous full response to maintain User Agent's own "thought" context
        messages.append({"role": "assistant", "content": previous_user_agent_full_response})

    blueprint_update_text = f"\nHere is the new or updated blueprint from the Planner:\n{updated_planner_blueprint}"
    messages.append({"role": "user", "content": blueprint_update_text})
    return messages


def generate_single_conversation_sample(challenge_topic_details, evaluation_config,
                                        challenge_axis_name, persona_seed,
                                        output_file_successful_break, output_file_passed_max_turns):
    """
    Generates a single multi-turn conversation sample based on the MultiChallenge methodology.
    """
    full_generation_log = []
    max_dialogue_turns = random.randint(5, 10) # As in original, e.g. 5 to 10 actual user-assistant pairs
    # max_iterations means number of user queries to generate.
    # So, if max_dialogue_turns is 5, we need 5 user queries and 5 assistant responses. Loop for 5 iterations.
    # The planner and user agent calls will happen *before* the user query for that iteration.

    current_planner_blueprint = None
    initial_planner_prompt_content = format_planner_prompt(
        PLANNER_BASE_PROMPT, challenge_axis_name, challenge_topic_details,
        evaluation_config, max_dialogue_turns, persona_seed
    )

    initial_user_agent_prompt_content = format_user_agent_prompt(
        USER_BASE_PROMPT, challenge_axis_name, "No blueprint available yet. Please wait for the first blueprint.",
        challenge_topic_details, max_dialogue_turns
    ) # Placeholder blueprint for first user agent message formatting

    direct_dialogue_turns = [] # Stores {"role": "user", "content": ...}, {"role": "assistant", ...}
    user_agent_last_full_response = None # To store the full output from user agent (thoughts, message, stop)

    full_generation_log.append({"type": "initial_setup", "details": {
        "challenge_topic": challenge_topic_details,
        "max_dialogue_turns_target": max_dialogue_turns,
        "persona": persona_seed,
        "challenge_axis": challenge_axis_name
    }})

    try:
        for turn_number in range(max_dialogue_turns): # Iterate for the number of desired user-assistant exchanges
            logging.info(f"--- Generating Turn {turn_number + 1} for axis '{challenge_axis_name}' ---")

            # 1. Planner Agent: Get/Update Blueprint
            if turn_number == 0:
                planner_messages = [{"role": "user", "content": initial_planner_prompt_content}]
            else:
                planner_messages = build_planner_update_messages(
                    initial_planner_prompt_content, # Send the full initial prompt so planner remembers its core task
                    current_planner_blueprint,
                    direct_dialogue_turns
                )

            full_generation_log.append({
                "type": "planner_agent_prompt", "turn": turn_number + 1,
                "content": planner_messages
            })
            current_planner_blueprint, planner_cot = get_openai_api_response_cot(msg=planner_messages)
            if not current_planner_blueprint:
                logging.error("Planner agent failed to return a blueprint.")
                return # or handle error appropriately
            full_generation_log.append({
                "type": "planner_agent_result", "turn": turn_number + 1,
                "blueprint": current_planner_blueprint, "thought_process": planner_cot
            })

            # Check if Planner decided to stop
            if "STOP THE CONVERSATION" in current_planner_blueprint.upper():
                logging.info(f"Planner signaled STOP at turn {turn_number + 1} based on its blueprint.")
                # User agent might still generate a final "STOP" message based on this
                # Or we can log this planner decision directly.
                # For now, let User Agent process this.
                pass


            # 2. User Agent: Generate User Query based on Blueprint
            if turn_number == 0 :
                 # For the very first user turn, the user agent only has the initial planner blueprint
                user_agent_messages = build_user_agent_update_messages(
                    initial_user_agent_prompt_content,
                    None, # No previous user agent response
                    current_planner_blueprint
                )
            else:
                user_agent_messages = build_user_agent_update_messages(
                    initial_user_agent_prompt_content, # Send its initial prompt for context
                    user_agent_last_full_response,
                    current_planner_blueprint
                )

            full_generation_log.append({
                "type": "user_agent_prompt", "turn": turn_number + 1,
                "content": user_agent_messages
            })
            user_agent_response_text, user_agent_cot = get_openai_api_response_cot(msg=user_agent_messages)
            if not user_agent_response_text:
                logging.error("User agent failed to return a response.")
                return
            user_agent_last_full_response = user_agent_response_text # Save full response

            user_query, stop_requested_by_user_agent = parse_user_agent_response(user_agent_response_text)
            full_generation_log.append({
                "type": "user_agent_result", "turn": turn_number + 1,
                "full_response": user_agent_response_text, "parsed_query": user_query,
                "stop_flag_parsed": stop_requested_by_user_agent, "thought_process": user_agent_cot
            })

            if stop_requested_by_user_agent:
                logging.info(f"User Agent signaled STOP at turn {turn_number + 1}. Saving to break file.")
                direct_dialogue_turns.append({"role": "user_agent_stop_signal", "content": user_query}) # Log the final message
                full_generation_log.append({"type": "final_state", "status": "stopped_by_agent", "dialogue": direct_dialogue_turns})
                with open(output_file_successful_break, "a", encoding="utf-8") as f:
                    f.write(json.dumps(full_generation_log, ensure_ascii=False, indent=2) + "\n")
                return # End generation for this sample

            direct_dialogue_turns.append({"role": "user", "content": user_query})

            # 3. Responder Agent (Simulated Target LLM): Get Assistant Response
            # For the responder, we only send the direct dialogue history.
            responder_messages = copy.deepcopy(direct_dialogue_turns)
            full_generation_log.append({
                "type": "responder_agent_prompt", "turn": turn_number + 1,
                "content": responder_messages
            })
            assistant_response, responder_cot = get_openai_api_response_cot(msg=responder_messages)
            if not assistant_response:
                logging.error("Responder agent failed to return a response.")
                # Potentially log this as a different kind of failure or retry
                direct_dialogue_turns.append({"role": "assistant", "content": "Error: Responder failed."})
                # Continue to let planner see this failure.
            else:
                direct_dialogue_turns.append({"role": "assistant", "content": assistant_response})

            full_generation_log.append({
                "type": "responder_agent_result", "turn": turn_number + 1,
                "response": assistant_response, "thought_process": responder_cot
            })

        # If loop completes without User Agent stopping:
        logging.info(f"Max turns ({max_dialogue_turns}) reached for axis '{challenge_axis_name}'. Saving to pass file.")
        full_generation_log.append({"type": "final_state", "status": "max_turns_reached", "dialogue": direct_dialogue_turns})
        with open(output_file_passed_max_turns, "a", encoding="utf-8") as f:
            f.write(json.dumps(full_generation_log, ensure_ascii=False, indent=2) + "\n")

    except Exception as e:
        logging.error(f"Error during generation for axis '{challenge_axis_name}', topic '{challenge_topic_details}': {e}", exc_info=True)
        full_generation_log.append({"type": "error", "message": str(e), "dialogue_so_far": direct_dialogue_turns})
        # Optionally save errored logs to a different file
        with open("errored_generations.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(full_generation_log, ensure_ascii=False, indent=2) + "\n")


if __name__ == '__main__':
    # Example for Inference Memory
    inference_topics = []
    for category, subtopics in Inference_Memory_Topic.items():
        for subtopic_description in subtopics:
            inference_topics.append({category: subtopic_description})

    selected_eval_config = Inference_Memory_Eval_Config
    selected_axis_name = ALL_CHALLENGE_AXES[1]  # "Inference Memory"

    # Shuffle for variety if running multiple times
    random.shuffle(inference_topics)
    random.shuffle(PERSONL_SEEDS_LIST)

    # For demonstration, generate for a few topics
    num_samples_to_generate_per_topic = 1 # Reduced for quick demo
    num_topics_to_try = 1 # Reduced for quick demo

    # Define output files
    output_broken_samples = "generated_broken_dialogues.jsonl"
    output_passed_samples = "generated_passed_dialogues.jsonl"

    # Clear files at start for a fresh run (optional)
    # open(output_broken_samples, 'w').close()
    # open(output_passed_samples, 'w').close()

    with ThreadPoolExecutor(max_workers=1) as executor: # Reduced workers for demo
        futures = []
        for topic_detail in inference_topics[:num_topics_to_try]:
            for i in range(num_samples_to_generate_per_topic):
                if not PERSONL_SEEDS_LIST:
                    logging.warning("No persona seeds available. Skipping generation.")
                    continue
                # Cycle through personas if fewer personas than samples needed
                current_persona = PERSONL_SEEDS_LIST[i % len(PERSONL_SEEDS_LIST)]
                logging.info(f"Submitting task for Axis: {selected_axis_name}, Topic: {topic_detail}, Persona: {current_persona[:50]}...")
                futures.append(executor.submit(
                    generate_single_conversation_sample,
                    topic_detail,
                    selected_eval_config,
                    selected_axis_name,
                    current_persona,
                    output_broken_samples,
                    output_passed_samples
                ))
        for future in futures:
            try:
                future.result()  # Wait for tasks to complete and catch exceptions
            except Exception as e:
                logging.error(f"A generation task failed: {e}", exc_info=True)

    logging.info("Data generation process finished.")
    logging.info(f"Broken samples saved to: {output_broken_samples}")
    logging.info(f"Passed (max turns) samples saved to: {output_passed_samples}")

    # --- Similar setup for Instruction Retention (example) ---
    # instruction_topics = []
    # for category, subtopics in Instruction_Retention_Topic.items():
    #     for subtopic_description in subtopics:
    #         instruction_topics.append({category: subtopic_description})
    #
    # selected_eval_config_instr = Instruction_Retention_Eval_Config
    # selected_axis_name_instr = ALL_CHALLENGE_AXES[0] # "Instruction Retention"
    # random.shuffle(instruction_topics)
    #
    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     futures_instr = []
    #     for topic_detail in instruction_topics[:num_topics_to_try]:
    #          for i in range(num_samples_to_generate_per_topic):
    #             if not PERSONL_SEEDS_LIST:
    #                 logging.warning("No persona seeds available. Skipping generation.")
    #                 continue
    #             current_persona = PERSONL_SEEDS_LIST[i % len(PERSONL_SEEDS_LIST)]
    #             logging.info(f"Submitting task for Axis: {selected_axis_name_instr}, Topic: {topic_detail}, Persona: {current_persona[:50]}...")
    #             futures_instr.append(executor.submit(
    #                 generate_single_conversation_sample,
    #                 topic_detail,
    #                 selected_eval_config_instr,
    #                 selected_axis_name_instr,
    #                 current_persona,
    #                 output_broken_samples, # Append to same files or use different ones
    #                 output_passed_samples
    #             ))
    #     for future in futures_instr:
    #         try:
    #             future.result()
    #         except Exception as e:
    #             logging.error(f"An instruction retention generation task failed: {e}", exc_info=True)