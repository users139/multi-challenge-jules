# 1. **Tone And Language**：
#    - 包括中立语气、正式语气和专业词汇。
# 2. **Response Structure**：
#    - 包括有限回应、包含特定元素和保持一致的回应格式。
# 3. **Grammar And Syntax**：
#    - 包括保持特定语法和嵌入特定词汇。
# 4. **Behavioral Consistency**：
#    - 包括始终同意用户的观点和作为客观信息源。
# 5. **Challenging Formats**：
#    - 包括诗歌式回应和指导模式。
axis = ["Instruction Retention","Inference Memory","Reliable Versioned Editing","Self-Coherence"]

Instruction_Retention_Topic = {
    "Tone And Language": [
        "Neutral Tone: Respond using a neutral, fact-based tone throughout the conversation, avoiding any opinions or subjective statements;",
        "Formal Tone: Respond using formal, structured language for the entire conversation, ensuring no informal or casual expressions are used;",
        "Specialized Vocabulary: Respond using vocabulary and terminology specific to a particular field (e.g., scientific, technical), maintaining this language style consistently"
    ],
    "Response Structure": [
        "Limited Responses: Respond with limited response types (e.g., yes/no, one-word answers), sticking to these brief answers without elaboration;",
        "Include Specific Element: Include a specific element (e.g., a number, name, or certain word) in each response, ensuring this feature is present in every reply;",
        "Consistent Format: Maintain a consistent response format (e.g., lists, questions, or structured paragraphs) for the entire conversation"
    ],
    "Grammar And Syntax": [
        "Maintain Specific Grammar: Respond using a specific grammatical rule or style (e.g., passive voice, present tense) throughout the conversation, ensuring no deviations;",
        "Embed Specific Words: Incorporate a particular word or set of words in every response, ensuring they are used naturally and grammatically in each sentence"
    ],
    "Behavioral Consistency": [
        "Agree With User: Respond by consistently agreeing with the user’s points or statements, avoiding any disagreement or contradiction;",
        "Objective Persona: Respond as an objective source of information (e.g., an encyclopedia), providing factual answers without any personal input or bias"
    ],
    "Challenging Formats": [
        "Poetic Response: Respond using a creative or structured format (e.g., rhyme, haiku, or metaphor) throughout the conversation, maintaining the format in each reply;",
        "Instructional Mode: Respond by providing step-by-step instructions or guides for the entirety of the conversation, regardless of the question’s complexity"
    ]
}


Inference_Memory_Topic = {
    "Personal Preference": [
        "Dietary Restrictions: The user mentions a dietary restriction (e.g., nut allergy, gluten intolerance) that the model must remember when making food or recipe suggestions.",
        "Favorite Cuisine: The user states their favorite type of cuisine (e.g., Italian, Japanese), and the model must refer to it when recommending a restaurant or dish.",
        "Specific Taste Preference: The user shares a dislike or preference for certain tastes (e.g., dislikes bitter foods), and the model must remember this when recommending future meals."
    ],
    "Schedule And Time": [
        "Event Date: The user mentions a specific date for an event, and the model must recall the date when helping plan activities or reminders.",
        "Time Conflict Management: The user mentions availability or a time conflict, and the model must adjust suggestions accordingly in future turns (e.g., offering an alternative time for a meeting).",
        "Recurring Event Recognition: The user mentions a recurring event (e.g., weekly meeting), and the model must implicitly refer to it when discussing future planning or scheduling."
    ],
    "Relationship Details": [
        "Personal Relationship Details: The user provides details about relationships (e.g., 'my girlfriend has a nut allergy'), and the model must recall these details when making suggestions or offering advice.",
        "Gift Preferences: The user states a preference for a particular gift idea, and the model must incorporate this when suggesting gift options later in the conversation.",
        "Anniversary Or Important Date: The user mentions a significant date (e.g., an anniversary), and the model must remember this when offering planning advice closer to that date."
    ],
    "Location And Travel": [
        "Travel Destination: The user mentions a planned travel destination, and the model must recall this when recommending activities or restaurants in that location.",
        "Distance Consideration: The user mentions distance preferences (e.g., prefers locations within walking distance), and the model must remember this when suggesting future plans or travel ideas.",
        "Previous Trip Comparison: The user recalls a past trip, and the model must implicitly refer back to it when comparing or suggesting similar travel destinations."
    ],
    "Health And Fitness": [
        "Fitness Goals: The user mentions fitness goals (e.g., weight loss, muscle building), and the model must recall these when suggesting workouts or routines.",
        "Health Conditions: The user mentions a health condition, and the model must take this into account when recommending fitness exercises or wellness activities.",
        "Recent Progress: The user talks about recent fitness progress, and the model should implicitly incorporate this into suggestions for future training plans."
    ],
    "Work And Project": [
        "Project Deadlines: The user mentions an important project deadline, and the model must recall it when providing advice on time management or setting reminders.",
        "Task Completion Status: The user updates the model on the completion status of a task, and the model should recollect this when suggesting next steps or priorities.",
        "Collaboration Details: The user mentions working with a specific colleague, and the model must refer to this detail when offering future advice or recommendations."
    ],
    "Learning And Development": [
        "Learning Goals: The user shares a learning goal (e.g., studying for an exam), and the model must recall this when suggesting study plans or materials.",
        "Recent Study Method: The user mentions a study method that worked well for them, and the model must incorporate this into future learning recommendations.",
        "Previous Learning Challenges: The user mentions difficulties with a topic, and the model must implicitly consider this when offering study tips or resource suggestions."
    ],
    "Hobbies And Interests": [
        "Hobby Details: The user shares a hobby or interest (e.g., photography), and the model must recall this when suggesting related activities or tools.",
        "Ongoing Project: The user mentions a personal project they are working on (e.g., building a model), and the model must refer to this when offering advice or ideas.",
        "Seasonal Activity Preference: The user shares a preference for a seasonal activity (e.g., skiing in winter), and the model must recollect this when the relevant season approaches."
    ],
    "Shopping And Purchases": [
        "Preferred Brands: The user states a preference for certain brands, and the model must remember this when suggesting products later.",
        "Previous Purchase Feedback: The model must recall this when recommending similar or related items.",
        "Price Sensitivity: The user mentions a specific budget or price preference, and the model must keep this in mind when making future shopping recommendations."
    ],
    "Entertainment And Media": [
        "Favorite Movies Or Shows: The user mentions favorite movies or TV shows, and the model should recall these when recommending new shows or movie nights.",
        "Music Preferences: The user states their music preferences, and the model must refer back to this when offering playlist or artist suggestions.",
        "Reading Habits: The user shares details about their reading habits (e.g., prefers mystery novels), and the model must recall this when recommending new books."
    ],
    "Tasks And Reminders": [
        "Task Details: The user provides details about a specific task, and the model must remember this when providing future reminders or check-ins.",
        "Reminder Modification: The user modifies a previously requested reminder, and the model must recollect the initial context and adjust the reminder accordingly.",
        "Chore Frequency: The user mentions how often they perform a chore (e.g., weekly laundry), and the model must recollect this when suggesting scheduling changes."
    ],
    "Emotional State": [
        "Emotional State: The user shares their emotional state or mood (e.g., feeling stressed), and the model must implicitly reference this in future supportive or calming suggestions.",
        "Mental Health Goals: The user mentions mental health goals (e.g., reducing anxiety), and the model must recall this when offering wellness or relaxation suggestions.",
        "Recent Emotional Experience: The user mentions a recent emotional experience (e.g., a stressful meeting), and the model must recollect this to offer empathy in future conversations."
    ],
    "Social And Cultural": [
        "Cultural Background: The user shares cultural details (e.g., food preferences based on culture), and the model must recollect this when making relevant suggestions.",
        "Social Event Details: The user mentions attending a social event, and the model must refer to this when discussing related future activities or follow-ups.",
        "Community Involvement: The user discusses involvement in a community or group, and the model must recall this when suggesting future activities or volunteering opportunities."
    ]
}


Reliable_Versioned_Editing_Topic = {
    "Technical": [
        "Code: Iteratively improve code by enhancing performance, fixing bugs, refactoring for readability, and adding error handling;",
        "Technical Documents: Improve technical documents by refining explanations, improving readability, and ensuring accuracy in complex information;",
        "Instruction Manuals: Improve instruction manuals by refining step-by-step directions, simplifying language, and enhancing usability;",
        "SOPs: Refine Standard Operating Procedures (SOPs) by improving clarity, ensuring compliance, and simplifying steps"
    ],
    "Writing And Content": [
        "Writing: Refine written content such as emails, reports, essays, or articles by improving clarity, tone, structure, and grammar;",
        "Creative Writing: Enhance creative works like stories, poems, or scripts by refining narrative flow, character development, pacing, and dialogue;",
        "Blog Posts: Improve blog posts by enhancing readability, refining structure, and aligning content with audience interests and SEO best practices;",
        "Speech Or Script: Refine speeches or scripts for events or media, improving pacing, tone, and effectiveness of key points"
    ],
    "Communication": [
        "Email Drafts: Improve email drafts by adjusting tone, clarity, and persuasiveness for professional, formal, or personal communications;",
        "Memos: Improve memos by refining tone, clarifying key points, and ensuring effective communication of information or directives;",
        "Customer Service Responses: Refine customer service responses by improving tone, ensuring clarity, and addressing customer concerns effectively"
    ],
    "Design And Presentation": [
        "Presentations: Enhance presentations by refining structure, flow, clarity of content, and visual appeal of slides;",
        "Website Copies: Improve website copy by enhancing readability, optimizing for SEO, and aligning messaging with target audience;",
        "Advertisement Copies: Improve ad copy for digital or print ads by refining messaging, enhancing call-to-action, and targeting specific audiences;",
        "Slogans Or Taglines: Enhance slogans or taglines by refining wording for memorability, clarity, and alignment with brand identity"
    ],
    "Planning And Strategy": [
        "Itineraries: Optimize travel or event itineraries, refining schedules, logistics, and activities to better suit preferences and goals;",
        "Project Plans: Refine project plans by improving task breakdown, timelines, resource allocation, and ensuring alignment with objectives;",
        "Budgets: Improve personal or business budgets by refining expense tracking, reallocating resources, and optimizing for financial goals;",
        "Business Proposals: Improve business proposals by refining structure, enhancing persuasive elements, and ensuring alignment with client needs"
    ],
    "Professional And Career": [
        "Resume Or CVs: Refine resumes or CVs to improve formatting, content structure, and highlight key skills or experiences more effectively;",
        "Job Descriptions: Enhance job descriptions by improving clarity, required qualifications, and ensuring alignment with role expectations;",
        "Resume Summaries: Improve the summary section of a resume, making it concise, impactful, and tailored to specific job roles;",
        "LinkedIn Profiles: Refine LinkedIn or other professional online profiles by enhancing descriptions, showcasing skills, and improving engagement"
    ],
    "Learning And Development": [
        "Study Plans: Refine study plans or schedules by optimizing time management, adjusting for learning goals, and improving content focus;",
        "Workshop Outlines: Enhance workshop outlines by refining structure, improving time allocation, and ensuring clarity in learning objectives;",
        "Workshop Material: Enhance workshop materials by improving content flow, ensuring alignment with learning objectives, and refining engagement techniques;",
        "Learning Modules: Improve educational learning modules by refining content structure, enhancing engagement, and aligning with learning outcomes"
    ],
    "Research And Documentation": [
        "Survey Or Forms: Enhance surveys or forms by improving question clarity, response options, and aligning with data collection goals;",
        "Grant Proposals: Refine grant proposals by enhancing the persuasiveness of the request, aligning with funding guidelines, and improving clarity;",
        "Financial Reports: Improve financial reports by refining clarity, highlighting key insights, and ensuring accurate data presentation;",
        "Public Policy Proposals: Refine policy proposals by improving clarity, structuring arguments logically, and addressing stakeholder concerns"
    ],
    "Customer And User Experience": [
        "Customer Surveys: Improve customer surveys by refining questions, ensuring clarity, and aligning with the goals of feedback collection;",
        "User Manual Or Guides: Improve user manuals or guides by refining instructions, adding clarity, and making them easier to navigate;",
        "Product Features: Refine product features by enhancing usability, improving design, and ensuring alignment with user feedback;",
        "FAQs: Refine Frequently Asked Questions (FAQ) sections by improving clarity, addressing common concerns, and simplifying explanations"
    ],
    "Event And Content Planning": [
        "Event Schedules: Refine event schedules by optimizing time management, improving logistics, and ensuring alignment with event goals;",
        "Webinar Content: Refine webinar content by improving clarity, pacing, and audience engagement strategies;",
        "Book Outlines: Improve book outlines by refining chapter structure, clarifying themes, and enhancing overall flow of ideas;",
        "Video Scripts: Improve video scripts by refining dialogue, enhancing pacing, and ensuring clear messaging for the intended audience"
    ]
}

Self_Coherence_Topic = {
    "Numerical Consistency": [
        "Enrollment Numbers: The model provides a specific enrollment number or capacity (e.g., 700 seats) and must maintain this consistency throughout the conversation without contradicting it later.",
        "Budget Or Cost Estimates: The model gives a financial estimate (e.g., a project costs $10,000) and must ensure this figure is consistently referenced in future statements.",
        "Time Calculations: The model provides a time estimate (e.g., 'the journey takes 5 hours') and must ensure it doesn’t give a conflicting estimate later in the conversation."
    ],
    "Fact Retention": [
        "Historical Dates: The model provides a specific date for a historical event (e.g., the signing of a treaty) and must avoid contradicting this date later in the conversation.",
        "Scientific Facts: The model offers a scientific fact (e.g., 'water boils at 100°C') and must ensure that this fact remains consistent if discussed again later.",
        "Geographical Details: The model provides information on a location (e.g., population of a city) and must maintain consistency when discussing similar data points."
    ],
    "Policy And Regulation Consistency": [
        "Law Or Policy Details: The model shares a policy or regulation (e.g., 'the legal drinking age is 18') and it must ensure consistency when discussing related regulations in future turns.",
        "Admission Requirements: The model specifies admission criteria (e.g., 'a score of 75% is required') and must avoid offering a different figure later when discussing eligibility.",
        "Legal Definitions: The model offers a legal definition and must ensure that future statements about the same concept or law align with the original explanation."
    ],
    "Personal Information Consistency": [
        "User Preferences: The model recalls specific user preferences mentioned earlier (e.g., preference for early morning workouts) and must maintain this information consistently throughout the conversation.",
        "Personal Details: The model provides information based on prior user input (e.g., a user’s preference for vegetarian meals) and must not contradict these details in later responses.",
        "Emotional State: The model acknowledges the user’s emotional state (e.g., feeling stressed) and must maintain empathy or appropriate responses without shifting tone unexpectedly."
    ],
    "Definition And Explanation Coherence": [
        "Term Definitions: The model provides a definition for a term (e.g., 'photosynthesis is the process by which plants convert light into energy') and must remain consistent if the term is referenced again.",
        "Explanation Of Concepts: The model explains a concept (e.g., 'gravity pulls objects toward the Earth') and must avoid contradicting or offering conflicting explanations later.",
        "Technical Jargon: The model offers technical details (e.g., 'HTTP stands for Hypertext Transfer Protocol') and must maintain consistency when discussing related topics."
    ],
    "Recommendation Consistency": [
        "Product Recommendations: The model suggests a specific product (e.g., 'you should buy a MacBook Pro') and must avoid contradicting this by recommending something entirely different later without context.",
        "Diet Or Health Advice: The model gives specific health or dietary advice (e.g., 'cut down on sugar') and must ensure it doesn’t recommend conflicting advice (e.g., 'eat more sweets') later.",
        "Travel Advice: The model suggests a travel itinerary or destination (e.g., 'visit Paris in the summer') and must not provide a conflicting recommendation (e.g., 'avoid Paris in the summer') later in the conversation."
    ],
    "Instruction And Process Consistency": [
        "Step By Step Instructions: The model provides a series of instructions (e.g., steps for setting up a device) and must ensure the sequence or details don’t change later in the conversation.",
        "Task Execution Details: The model explains how to execute a task (e.g., 'first open the settings menu') and must maintain consistency when reiterating the steps later.",
        "Safety Instructions: The model offers safety instructions (e.g., 'wear protective gloves while handling chemicals') and must ensure consistency when discussing safety measures in future interactions."
    ],
    "Mathematical Coherence": [
        "Equation Consistency: The model provides a mathematical solution or formula (e.g., 'the area of a circle is πr²') and must maintain the same formula if referenced again later.",
        "Financial Calculations: The model performs a financial calculation (e.g., calculating interest or budgets) and must ensure the figures remain consistent when the topic is revisited.",
        "Unit Conversions: The model converts units (e.g., miles to kilometers) and must ensure the same conversion factors are used consistently throughout the conversation."
    ],
    "Contextual Coherence": [
        "Cultural References: The model references a cultural event or norm (e.g., 'Thanksgiving is celebrated in November') and must not contradict this fact later in the conversation.",
        "Language And Tone: The model adopts a specific tone or style (e.g., formal, friendly) and must maintain that tone consistently without abruptly switching styles unless prompted.",
        "Scenario Consistency: The model establishes a scenario (e.g., 'you are planning a weekend getaway') and must avoid changing key details of the scenario in later responses (e.g., suggesting activities for a weekday instead)."
    ],
    "Ethical And Moral Consistency": [
        "Ethical Advice: The model provides ethical advice (e.g., 'it’s important to respect privacy') and must avoid offering conflicting or hypocritical advice (e.g., suggesting invasive actions) later.",
        "Moral Judgments: The model makes a moral judgment (e.g., 'lying is wrong') and must ensure future advice remains aligned with that judgment and does not contradict it.",
        "Legal Or Ethical Conflicts: The model advises following the law or ethical guidelines (e.g., 'never drink and drive') and must ensure that no later suggestions contradict this advice."
    ],
    "Story Or Narrative Consistency": [
        "Character Details: In a storytelling or narrative context, the model introduces a character with certain traits (e.g., 'John is an honest man') and must not contradict this by later describing the character as dishonest.",
        "Plot Details: The model provides specific plot points in a narrative (e.g., 'the treasure is hidden in the mountains') and must ensure these details remain consistent throughout the story.",
        "World-Building Facts: In creative writing or world-building, the model creates rules or facts about the world (e.g., 'magic doesn’t exist in this universe') and must not introduce contradictions (e.g., introducing magic) later."
    ],
    "Opinion Consistency": [
        "Opinion On Products: The model expresses an opinion on a product or service (e.g., 'I think this phone is excellent') and must ensure consistency if the same product is discussed again.",
        "Taste Or Preference: The model expresses a preference (e.g., 'chocolate is the best dessert') and must avoid contradicting this by stating a different preference (e.g., 'ice cream is the best') later without context.",
        "Political Or Social Views: The model shares a political or social opinion (e.g., 'climate change is a serious issue') and must remain consistent in related discussions throughout the conversation."
    ]
}


Instruction_Retention_Eval_Config = {
    "category_name": "Instruction Retention",
    "category_definition": "This challenge category evaluates the model’s ability to maintain a SPECIFIC instruction throughout the entire dialogue. It covers the model’s ability to sustain a specific instruction over a wide variety of user questions. This is ALWAYS in the first user message and the model MUST be informed explicitly that the instruction is for the ENTIRE conversation.",
    "pass_criteria": "The model consistently adheres to the established instruction or guideline throughout the conversation without deviation.",
    "failure_criteria": "If the model doesn’t adhere to the initial instruction in any way this would be considered a failure on this metric.",
    "example": "The user instructs the model in the first turn to avoid expressing any opinions for the entirety of the conversation. The user tests the model by discussing topics like preferences in food, sports, and politics. The assistant remains neutral throughout, but in the final turn, it fails by stating, 'You’ve certainly engaged in a thought-provoking and interesting conversation,' when the user asked the model to rate their conversational skills."
}

Inference_Memory_Eval_Config = {
    "category_name": "Inference Memory",
    "category_definition": "This metric evaluates the model’s ability to retain and accurately reference SPECIFIC information from previous turns in the conversation, especially from multiple turns ago. The focus is on how well the model can remember SPECIFIC details, facts, or topics that were discussed earlier in the dialogue and bring them up when relevant in later stages of the conversation.",
    "pass_criteria": "The model successfully recalls and integrates the necessary context from earlier turns, making its responses coherent and contextually appropriate.",
    "failure_criteria": "If the model shows any indication that it forgot or misremembered key details from previous turns, leading to incoherent or contextually inconsistent responses, it would be considered a failure on this challenge category.",
    "example": "In a conversation about a dinner date, the user mentions their girlfriend is allergic to nuts. In a later turn, when the user asks for food recommendations, the model suggests dishes with nuts as ingredients, forgetting the allergy mentioned earlier."
}

Reliable_Versioned_Editing_Eval_Config = {
    "category_name": "Reliable Versioned Editing",
    "category_definition": "This challenge category evaluates the model’s ability to process, integrate, and adapt to multiple, evolving instructions over the course of a conversation. Reliable Version Editing tests the model’s flexibility and capacity to evolve its responses in accordance with increasingly complex or layered directives.",
    "pass_criteria": "The model demonstrates a clear ability to refine its answers, successfully integrating multiple, evolving instructions while maintaining coherence. It should offer a response that respects all directives given, ensuring that no earlier instruction is forgotten, contradicted, or unduly deprioritized unless explicitly told to do so.",
    "failure_criteria": "If the model fails to combine and respect the evolving instructions — for example, by ignoring a previous directive in favor of a newer one, contradicting earlier responses, or offering an answer that doesn’t fully accommodate all the instructions provided — this would be considered a failure on the reliable version editing challenge category.",
    "example": "The user initially asks the model to help plan a week-long vacation itinerary for Italy. After the model suggests an itinerary, the user then requests adjustments, such as adding a specific restaurant in Rome, increasing time spent at museums, and removing one destination in favor of another. If the final itinerary fails to incorporate one or more of the requested modifications, this would be considered a failure on the reliable version editing challenge category."
}

Self_Coherence_Eval_Config = {
    "category_name": "Self-Coherence",
    "category_definition": "This challenge category specifically evaluates the model’s ability to avoid contradictions in its responses. As new information or instructions are introduced, the model must maintain internal consistency, ensuring that its answers do not conflict with prior statements.",
    "pass_criteria": "The model successfully avoids contradictions in its responses. It provides consistent information throughout the conversation, ensuring that new responses do not conflict with earlier ones.",
    "failure_criteria": "The model fails if it provides contradictory information during the conversation based on its own limitations, not as a result of user manipulation or gaslighting.",
    "example": "The user asks the model about the distance from Earth to the Moon, and the model correctly responds with approximately 384,400 kilometers. Later in the conversation, when asked to explain how far light travels from the Earth to the Moon in one second, the model incorrectly states that the distance is only 300,000 kilometers instead of maintaining internal consistency with the previous answer."
}


def load_json(json_file):
    res = []
    with open(json_file, 'r', encoding='utf-8') as file:
        data = file.readlines()
        for s_data in data:
            try:
                s_data = json.loads(s_data)
                res.append(s_data)
            except:
                print("expected",s_data)

    return res

def get_person_seed_list():
    ori_person = load_json("persona.jsonl")
    # print()
    p_list = [d["persona"] for d in ori_person]
    return p_list

PERSONL_SEEDS_LIST = get_person_seed_list()

if __name__ == '__main__':
    personl = get_person_seed_list()

    print(len(personl))