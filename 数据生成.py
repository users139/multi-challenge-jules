# main_refactored_full_generation.py

import copy
import json
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os # For creating output directory

# Assuming r1.py is in the same directory or accessible in PYTHONPATH
from r1 import get_openai_api_response_cot # 假设 r1.py 和 get_openai_api_response_cot 可用

from ours.from_yuxian_chinese_openrouter_new_prompts.config import (
    axis as ALL_CHALLENGE_AXES,
    Instruction_Retention_Topic, Inference_Memory_Topic,
    Reliable_Versioned_Editing_Topic, Self_Coherence_Topic,
    Instruction_Retention_Eval_Config, Inference_Memory_Eval_Config,
    Reliable_Versioned_Editing_Eval_Config, Self_Coherence_Eval_Config,
    PERSONL_SEEDS_LIST
)
from planer import prompt as PLANNER_BASE_PROMPT
from user import prompt as USER_BASE_PROMPT

# Setup logging for better traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

MAX_CONVERSATION_HISTORY_TURNS_FOR_PLANNER = 10
OUTPUT_DIRECTORY = "generated_data_all_topics"

# --- (此处粘贴之前重构代码中的辅助函数) ---
# format_planner_prompt
# format_user_agent_prompt
# parse_user_agent_response
# build_conversation_history_text
# build_planner_update_messages
# build_user_agent_update_messages
# generate_single_conversation_sample (确保此函数中的输出文件名参数被正确使用或修改为接受目录)

# (为了简洁，我将假设这些辅助函数已在此处定义，与您之前提供的重构版本一致)
# START - 辅助函数定义 (从之前的重构版本复制)
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
    prompt = base_prompt_template.replace("{axis}", challenge_axis)
    prompt = prompt.replace("{topic}", json.dumps(topic_details, indent=4, ensure_ascii=False))
    prompt = prompt.replace("{num_turns}", str(max_dialogue_turns))
    prompt = prompt.replace("{blueprint}", current_blueprint)
    return prompt

def parse_user_agent_response(response_text):
    """Parses the User Agent's structured response to extract message and stop flag."""
    import re
    user_message = None
    stop_flag = False

    msg_match = re.search(r'User Message:\*?\*?\s*"?([^"\n]*?)"?\s*(\n|$)', response_text, re.DOTALL | re.IGNORECASE)
    if not msg_match:
        msg_match = re.search(r'User Message:\s*(.*?)(\n•\s*STOP:|$)', response_text, re.DOTALL | re.IGNORECASE)

    if msg_match:
        user_message = msg_match.group(1).strip()
    else:
        logging.warning(f"Could not extract User Message from: {response_text}")
        user_message = f"Error: Could not parse user message from original: {response_text}"

    stop_match = re.search(r'STOP:\*?\*?\s*(True|False)', response_text, re.IGNORECASE)
    if stop_match:
        stop_flag_str = stop_match.group(1).strip().lower()
        stop_flag = stop_flag_str == 'true'
    else:
        logging.warning(f"Could not extract STOP flag from: {response_text}, defaulting to False.")

    return user_message, stop_flag

def build_conversation_history_text(direct_dialogue_turns):
    """Builds a text representation of the conversation history for the planner."""
    history_text = "\nHere is the conversation history up to the last assistant response:\n"
    start_index = max(0, len(direct_dialogue_turns) - MAX_CONVERSATION_HISTORY_TURNS_FOR_PLANNER * 2)
    for i in range(start_index, len(direct_dialogue_turns)):
        turn = direct_dialogue_turns[i]
        history_text += f"{turn['role'].capitalize()}: {turn['content']}\n"
    return history_text.strip()

def build_planner_update_messages(initial_planner_prompt_content, previous_planner_blueprint,
                                  current_direct_dialogue_turns):
    """Constructs the message list for updating the planner's blueprint."""
    messages = [{"role": "user", "content": initial_planner_prompt_content}]
    if previous_planner_blueprint:
        messages.append({"role": "assistant", "content": previous_planner_blueprint})
    messages.append({"role": "user", "content": build_conversation_history_text(current_direct_dialogue_turns)})
    return messages

def build_user_agent_update_messages(initial_user_agent_prompt_content, previous_user_agent_full_response,
                                     updated_planner_blueprint):
    """Constructs the message list for the user agent to generate the next turn."""
    messages = [{"role": "user", "content": initial_user_agent_prompt_content}]
    if previous_user_agent_full_response:
        messages.append({"role": "assistant", "content": previous_user_agent_full_response})
    blueprint_update_text = f"\nHere is the new or updated blueprint from the Planner:\n{updated_planner_blueprint}"
    messages.append({"role": "user", "content": blueprint_update_text})
    return messages

def generate_single_conversation_sample(challenge_topic_details, evaluation_config,
                                        challenge_axis_name, persona_seed,
                                        output_dir): # 修改为接受输出目录
    """
    Generates a single multi-turn conversation sample.
    Outputs to files named based on axis and status within the output_dir.
    """
    # 为每个样本创建单独的日志记录，避免混淆
    sample_log_id = f"{challenge_axis_name.replace(' ', '_')}_{list(challenge_topic_details.keys())[0].replace(' ', '_')}_{random.randint(1000,9999)}"
    logging.info(f"[{sample_log_id}] Starting generation...")

    full_generation_log = []
    max_dialogue_turns = random.randint(5, 8) # 稍微减少最大轮数以加速示例，可调整回 5, 10
    current_planner_blueprint = None
    initial_planner_prompt_content = format_planner_prompt(
        PLANNER_BASE_PROMPT, challenge_axis_name, challenge_topic_details,
        evaluation_config, max_dialogue_turns, persona_seed
    )
    initial_user_agent_prompt_content = format_user_agent_prompt(
        USER_BASE_PROMPT, challenge_axis_name, "No blueprint available yet.",
        challenge_topic_details, max_dialogue_turns
    )
    direct_dialogue_turns = []
    user_agent_last_full_response = None

    full_generation_log.append({"type": "initial_setup", "sample_id": sample_log_id, "details": {
        "challenge_topic": challenge_topic_details,
        "max_dialogue_turns_target": max_dialogue_turns,
        "persona": persona_seed,
        "challenge_axis": challenge_axis_name
    }})

    output_file_successful_break = os.path.join(output_dir, f"{challenge_axis_name.replace(' ', '_')}_broken.jsonl")
    output_file_passed_max_turns = os.path.join(output_dir, f"{challenge_axis_name.replace(' ', '_')}_passed.jsonl")
    output_file_errored = os.path.join(output_dir, f"{challenge_axis_name.replace(' ', '_')}_errored.jsonl")

    try:
        for turn_number in range(max_dialogue_turns):
            logging.info(f"[{sample_log_id}] Turn {turn_number + 1}/{max_dialogue_turns}")

            # 1. Planner Agent
            if turn_number == 0:
                planner_messages = [{"role": "user", "content": initial_planner_prompt_content}]
            else:
                planner_messages = build_planner_update_messages(
                    initial_planner_prompt_content,
                    current_planner_blueprint,
                    direct_dialogue_turns
                )
            full_generation_log.append({"type": "planner_agent_prompt", "turn": turn_number + 1, "content": planner_messages})
            current_planner_blueprint, planner_cot = get_openai_api_response_cot(msg=planner_messages)
            if not current_planner_blueprint:
                raise ValueError("Planner agent returned empty blueprint.")
            full_generation_log.append({"type": "planner_agent_result", "turn": turn_number + 1, "blueprint": current_planner_blueprint, "thought_process": planner_cot})

            # 2. User Agent
            if turn_number == 0 :
                user_agent_messages = build_user_agent_update_messages(initial_user_agent_prompt_content, None, current_planner_blueprint)
            else:
                user_agent_messages = build_user_agent_update_messages(initial_user_agent_prompt_content, user_agent_last_full_response, current_planner_blueprint)

            full_generation_log.append({"type": "user_agent_prompt", "turn": turn_number + 1, "content": user_agent_messages})
            user_agent_response_text, user_agent_cot = get_openai_api_response_cot(msg=user_agent_messages)
            if not user_agent_response_text:
                raise ValueError("User agent returned empty response.")
            user_agent_last_full_response = user_agent_response_text
            user_query, stop_requested_by_user_agent = parse_user_agent_response(user_agent_response_text)
            full_generation_log.append({"type": "user_agent_result", "turn": turn_number + 1, "full_response": user_agent_response_text, "parsed_query": user_query, "stop_flag_parsed": stop_requested_by_user_agent, "thought_process": user_agent_cot})

            if stop_requested_by_user_agent:
                logging.info(f"[{sample_log_id}] User Agent signaled STOP. Saving to break file.")
                direct_dialogue_turns.append({"role": "user_agent_stop_signal", "content": user_query if user_query else "User agent stopped with no message."})
                full_generation_log.append({"type": "final_state", "status": "stopped_by_agent", "dialogue": direct_dialogue_turns})
                with open(output_file_successful_break, "a", encoding="utf-8") as f:
                    f.write(json.dumps(full_generation_log, ensure_ascii=False, indent=2) + "\n")
                return

            if not user_query or "Error: Could not parse user message" in user_query: # 如果无法解析出有效的用户查询
                logging.warning(f"[{sample_log_id}] User agent produced an unparsable/empty query. Ending this sample as errored.")
                raise ValueError(f"User agent produced unparsable/empty query: {user_agent_response_text}")


            direct_dialogue_turns.append({"role": "user", "content": user_query})

            # 3. Responder Agent
            responder_messages = copy.deepcopy(direct_dialogue_turns)
            full_generation_log.append({"type": "responder_agent_prompt", "turn": turn_number + 1, "content": responder_messages})
            assistant_response, responder_cot = get_openai_api_response_cot(msg=responder_messages)
            if not assistant_response: # 允许空响应，但记录
                logging.warning(f"[{sample_log_id}] Responder agent returned empty response.")
                assistant_response = "[Responder Agent returned empty or failed]"
            direct_dialogue_turns.append({"role": "assistant", "content": assistant_response})
            full_generation_log.append({"type": "responder_agent_result", "turn": turn_number + 1, "response": assistant_response, "thought_process": responder_cot})

            # 检查 Planner 是否在蓝图中指示停止 (在User Agent处理之前，这里也检查一下)
            if "STOP THE CONVERSATION" in current_planner_blueprint.upper():
                 logging.info(f"[{sample_log_id}] Planner's blueprint indicated STOP after assistant's response. This sample may be considered 'broken' based on planner's final assessment.")
                 # 这里可以根据具体逻辑决定是否立即判定为 broken 并保存
                 # 为了与原逻辑更接近（User Agent最终决定），我们让User Agent在下一轮处理这个STOP
                 # 但如果Planner的STOP是最终决定，这里可以直接保存到broken文件

        logging.info(f"[{sample_log_id}] Max turns reached. Saving to pass file.")
        full_generation_log.append({"type": "final_state", "status": "max_turns_reached", "dialogue": direct_dialogue_turns})
        with open(output_file_passed_max_turns, "a", encoding="utf-8") as f:
            f.write(json.dumps(full_generation_log, ensure_ascii=False, indent=2) + "\n")

    except Exception as e:
        logging.error(f"[{sample_log_id}] Error during generation: {e}", exc_info=True)
        full_generation_log.append({"type": "error", "message": str(e), "dialogue_so_far": direct_dialogue_turns})
        with open(output_file_errored, "a", encoding="utf-8") as f:
            f.write(json.dumps(full_generation_log, ensure_ascii=False, indent=2) + "\n")

# END - 辅助函数定义

if __name__ == '__main__':
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    logging.info(f"Ensuring output directory exists: {OUTPUT_DIRECTORY}")

    # 定义所有挑战的配置映射
    challenge_configs = {
        ALL_CHALLENGE_AXES[0]: (Instruction_Retention_Topic, Instruction_Retention_Eval_Config),
        ALL_CHALLENGE_AXES[1]: (Inference_Memory_Topic, Inference_Memory_Eval_Config),
        ALL_CHALLENGE_AXES[2]: (Reliable_Versioned_Editing_Topic, Reliable_Versioned_Editing_Eval_Config),
        ALL_CHALLENGE_AXES[3]: (Self_Coherence_Topic, Self_Coherence_Eval_Config),
    }

    tasks_to_submit = []
    # 为每个挑战轴的每个主题的每个子主题描述创建一个任务
    # 可以调整 num_samples_per_subtopic 来控制每个最细粒度主题生成的样本数
    num_samples_per_subtopic = 1 # 每个子主题生成1个样本用于演示，可以调大

    if not PERSONL_SEEDS_LIST:
        logging.error("PERSONL_SEEDS_LIST is empty. Cannot generate data without personas.")
        exit()

    persona_iterator = 0 # 用于循环使用 persona

    for axis_name, (topic_map, eval_config) in challenge_configs.items():
        logging.info(f"Preparing tasks for Axis: {axis_name}")
        for topic_category, subtopic_descriptions in topic_map.items():
            for subtopic_desc in subtopic_descriptions:
                current_topic_detail = {topic_category: subtopic_desc}
                for _ in range(num_samples_per_subtopic):
                    # 循环使用 Persona 种子
                    current_persona = PERSONL_SEEDS_LIST[persona_iterator % len(PERSONL_SEEDS_LIST)]
                    persona_iterator += 1

                    tasks_to_submit.append({
                        "challenge_topic_details": current_topic_detail,
                        "evaluation_config": eval_config,
                        "challenge_axis_name": axis_name,
                        "persona_seed": current_persona,
                        "output_dir": OUTPUT_DIRECTORY
                    })
    random.shuffle(tasks_to_submit) # 打乱任务顺序，避免集中请求相似主题

    logging.info(f"Total tasks to submit: {len(tasks_to_submit)}")

    # 调整线程池大小，请根据您的API限制和机器性能调整
    # 大量任务并发可能导致API限流
    max_concurrent_workers = 5 # 示例值
    completed_tasks = 0
    failed_tasks = 0

    with ThreadPoolExecutor(max_workers=max_concurrent_workers) as executor:
        futures = {executor.submit(
                        generate_single_conversation_sample,
                        **task_args
                   ): task_args for task_args in tasks_to_submit}

        for future in as_completed(futures):
            task_info = futures[future]
            try:
                future.result()  # Wait for task to complete, will raise exception if task did
                logging.info(f"Task completed for topic: {task_info['challenge_topic_details']} on axis: {task_info['challenge_axis_name']}")
                completed_tasks +=1
            except Exception as e:
                logging.error(f"Task failed for topic: {task_info['challenge_topic_details']} on axis: {task_info['challenge_axis_name']}. Error: {e}", exc_info=False) # exc_info=False to avoid huge traceback spam for many errors
                failed_tasks +=1
            logging.info(f"Progress: {completed_tasks+failed_tasks}/{len(tasks_to_submit)} tasks processed. (Completed: {completed_tasks}, Failed: {failed_tasks})")


    logging.info("----------------------------------------------------")
    logging.info("All data generation tasks process attempt finished.")
    logging.info(f"Total tasks submitted: {len(tasks_to_submit)}")
    logging.info(f"Successfully completed tasks (may include 'passed' or 'broken' saves): {completed_tasks}")
    logging.info(f"Tasks that raised an exception during processing: {failed_tasks}")
    logging.info(f"Generated data saved in directory: {OUTPUT_DIRECTORY}")
    logging.info("Please check the .jsonl files in the output directory.")
    logging.info("----------------------------------------------------")