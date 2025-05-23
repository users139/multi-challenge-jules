import copy
import json
import random

from planer import prompt as plan_prompt_ori
from user import prompt as user_prompt_ori
from r1 import get_openai_api_response_cot
# 生成会话策略
from ours.from_yuxian_chinese_openrouter_new_prompts.config import axis
from ours.from_yuxian_chinese_openrouter_new_prompts.config import Inference_Memory_Topic
from ours.from_yuxian_chinese_openrouter_new_prompts.config import \
    Inference_Memory_Eval_Config
from ours.from_yuxian_chinese_openrouter_new_prompts.config import PERSONL_SEEDS_LIST
print(len(PERSONL_SEEDS_LIST))






def generate_data(topic,eval_config,axis):
    personl_seed = random.sample(PERSONL_SEEDS_LIST,1)[0]
    def get_planer_prompt(plan_prompt, axis, topic,eval_config,max_turns):
        plan_prompt = plan_prompt.replace("{axis}", axis)
        plan_prompt = plan_prompt.replace("{topic}", json.dumps(topic, indent=4))
        plan_prompt = plan_prompt.replace("{num_turns}", max_turns)
        plan_prompt = (plan_prompt.replace("{Definition}", eval_config["category_definition"]).
                       replace("{Pass Criteria}", eval_config["pass_criteria"]).
                       replace("{Fail Criteria}", eval_config["failure_criteria"]).
                       replace("{Example}", eval_config["example"]).replace("{Persona_seed}", personl_seed))
        return plan_prompt

    def get_user_prompt(user_prompt, axis, blueprint,topic,max_turns):
        user_prompt = user_prompt.replace("{axis}", axis[0])
        user_prompt = user_prompt.replace("{topic}", json.dumps(topic, indent=4))
        user_prompt = user_prompt.replace("{num_turns}", max_turns)
        user_prompt = user_prompt.replace("{blueprint}", blueprint)
        return user_prompt

    def get_user_query(text):
        import re
        if "**User Message:** " in text:
            match = re.search(r'\*\*User Message:\*\* "(.*?)"', text)
        else:
            match = re.search(r'\*\*User Message\*\*: (.*?)', text)
        if match:
            print("\n使用正则表达式提取的结果query:")
            print(match.group(1))
        else:
            print(text)
        return match.group(1)

    def get_user_stop(text):
        import re
        if "**STOP:**" in text:
            # print("hahaha")
            match = re.search(r'\*\*STOP:\*\*\s*(True|False)', text)
        if "**STOP**:" in text:
            match = re.search(r'\*\*STOP\*\*:\s*(True|False)', text)
        if match:
            print("\n使用正则表达式提取的结果BOOL:")
            print(match.group(1))
        else:
            print(text)
        return match.group(1)

    conversation_history = f"""
    Here is the conversation history:
    """
    def refresh_conversation_history(conversation_history,user,assistant):
        conversation_history += f"""
        User:{user}
        Assistant:{assistant}
        """
        return conversation_history

    def refresh_blueprint_history(blueprint):
        blueprint_history = f"""
        Here is the new blueprint:
        {blueprint}
        """
        return blueprint_history

    def save_one_sample(i,text,max_turns):
        print("save one")
        print(text)
        is_save = False
        if "true" == get_user_stop(text).lower():
            print("STOP THE CONVERSATION RIGHT NOW.....结束啦！")
            all_message_record.append(
                {"role": f"stop_flag", "content": "STOP THE CONVERSATION RIGHT NOW"})
            with open("stop_data_result.json","a",encoding="utf-8") as f:
                f.write(json.dumps(all_message_record,ensure_ascii=False)+"\n")
            is_save = True
            return is_save
        if i==max_turns:
            print("save")
            all_message_record.append(
                {"role": f"stop_flag", "content": "The conversation is complete!"})
            with open("pass_data.json","a",encoding="utf-8") as f:
                f.write(json.dumps(all_message_record,ensure_ascii=False)+"\n")
            is_save = True
            return is_save
            # break
        return is_save

    all_message_record = []
    max_turns = random.randint(5,10)
    msg_for_diag = []
    all_message_record.append({"topic":topic})
    all_message_record.append({"预估_max_turns": max_turns})
    all_message_record.append({"personal": personl_seed})
    try:
        for i in range(max_turns+1):
            plan_prompt = copy.deepcopy(plan_prompt_ori)
            user_prompt = copy.deepcopy(user_prompt_ori)
            if i == 0:
                # inint PLAN Agent
                plan_prompt = get_planer_prompt(plan_prompt, axis, topic,eval_config,str(max_turns))
                blueprint, cot = get_openai_api_response_cot(msg = [{"role":"user","content":plan_prompt}], max_retries=5)
                all_message_record.append({"role": f"Plan-Agent-Prompt_{i}", "content": plan_prompt})
                all_message_record.append({"role": f"Plan-Agent-Result_{i}","content": blueprint,"think":cot})
                # User Agent
                # print(user_prompt)
                user_prompt = get_user_prompt(user_prompt, axis, blueprint, topic ,str(max_turns))
                msg_for_user_res_list = [{"role":"user","content":user_prompt}]
            else:
                msg_for_user_res_list.append({"role": "assistant", "content": user_res})
                msg_for_user_res_list.append({"role": "user", "content": refresh_blueprint_history(blueprint)})
            user_res, cot = get_openai_api_response_cot(msg = msg_for_user_res_list, max_retries=5)
            # print(user_res)
            # all_message_record.append({"role": f"User-Agent-Prompt_{i}", "content": user_prompt})
            all_message_record.append({"role": f"User-Agent-Result_更新_{i}", "content": user_res, "think": cot})
            is_save = save_one_sample(i, user_res, max_turns)
            if is_save:
                break
            # print(user_res)
            query = get_user_query(user_res)
            # Response Agent
            # print("query:",query)
            msg_for_diag.append({"role":"user","content":query})
            assistant_res, cot = get_openai_api_response_cot(msg = msg_for_diag, max_retries=5)
            msg_for_diag.append({"role": "assistant", "content": assistant_res})
            # print(assistant_res)
            all_message_record.append({"role": f"用户-Query（真实对话）_{i}", "content": query})
            all_message_record.append({"role": f"助手-Result（真实对话）_{i}", "content": assistant_res,"think": cot})

            conversation_history = refresh_conversation_history(conversation_history,query,assistant_res)

            # Please update your conversation blueprint.
            if i == 0:
                # 初始化
                msg_for_plan = [{"role": "user","content":plan_prompt}]
            msg_for_plan.append({"role": "assistant","content":blueprint})
            msg_for_plan.append({"role": "user", "content": conversation_history})
            print("cur msg_for_plan")
            print(msg_for_plan)
            blueprint, cot = get_openai_api_response_cot(msg = msg_for_plan, max_retries=5)

            all_message_record.append({"role": f"Plan-Agent-Prompt-更新_{i}", "content": copy.deepcopy(msg_for_plan)})
            all_message_record.append({"role": f"Plan-Agent-Result-更新_{i}", "content": copy.deepcopy(blueprint), "think": cot})
    except:
        print("error>>>>>>>")
        return

if __name__ == '__main__':

    # topic_list = []
    # for k,v in Instruction_Retention_Topic.items():
    #     for detail in v:
    #         topic_list.append({k:detail})
    # # topic = {"Tone And Language": Instruction_Retention_Topic["Tone And Language"][1]}
    # eval_config = Instruction_Retention_Eval_Config
    # axis = axis[0]
    # random.shuffle(topic_list)
    # for topic in topic_list:
    #     # generate_data(topic, eval_config, axis)
    #     from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    #     num_process = 5
    #     with ThreadPoolExecutor(max_workers=num_process) as pool_1:
    #         tasks = []
    #         for _ in range(50):
    #             tasks.append(pool_1.submit(generate_data,
    #                                        topic,
    #                                        eval_config,
    #                                        axis
    #                                        ))

    topic_list = []
    for k, v in Inference_Memory_Topic.items():
        for detail in v:
            topic_list.append({k: detail})
    # topic = {"Tone And Language": Instruction_Retention_Topic["Tone And Language"][1]}
    eval_config = Inference_Memory_Eval_Config
    axis = axis[1]
    random.shuffle(topic_list)
    for topic in topic_list:
        # generate_data(topic, eval_config, axis)
        from concurrent.futures import ThreadPoolExecutor

        num_process = 1
        with ThreadPoolExecutor(max_workers=num_process) as pool_1:
            tasks = []
            for _ in range(1):
                tasks.append(pool_1.submit(generate_data,
                                           topic,
                                           eval_config,
                                           axis
                                           ))