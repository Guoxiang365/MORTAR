from groq import Groq

LLM_USAGE_COUNTER = 0
LLM_TOKEN_COUNTER = 0

client = Groq(
    api_key="Your Groq API Key",
)

GENERAL_SYSTEM_PROMPT = "You are an expert in linguistics. You will help the user to finish their task precisely. Do not give any explanation and apologize."

def extract_declerative_information(dialogue_content, return_raw=False, llm_temperature=0):
    EXTRACT_DECLEARATIVE_INFORMATION_PROMPT = """
    Please help me summarize the trunk information confirmed in each round of the dialogue with a declarative sentence. 
    Please make anaphora resolution and do not use any pronouns in your output, so that each piece of information is independent.
    Do not change the original name of any entities.
    Output in JSON format: {{"Round 1":<information confirmed in Round 1>, "Round 2":<information confirmed in Round 2>, ...}}
    ======================================================================
    Dialogue:
    {dialogue_content}
    Output:
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": GENERAL_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": EXTRACT_DECLEARATIVE_INFORMATION_PROMPT.format(dialogue_content=dialogue_content)
            }
        ],
        model="llama3-70b-8192",
        response_format={"type": "json_object"},
        temperature=llm_temperature,
        max_tokens=8192,
        top_p=1,
        stream=False,
        stop=None,
    )
    returned_content = chat_completion.choices[0].message.content

    if return_raw:
        return returned_content
    else:
        dict_round_information = eval(returned_content)
        return dict_round_information


def question_resolution(dialogue_content, return_raw=False, llm_temperature=0):
    EXTRACT_FULL_INFO_QUESTION_PROMPT = """
Task: Extract coreference information and rewrite each Question and Answer in the provided Dialogue_history so that each Question does not contain any grammatical ellipsis.

Output in JSON format: {{"Round 1":{{"Question": <information_filled_Question>, "Answer": <information filled Answer>}}, "Round 2":...}}

=====================================================================
Example:
=====================================================================
Dialogue_history:
{{"Round 1":{{"Question": "what was the name of the fish?", "Answer": "Asta."}}, "Round 2":{{"Question": "What looked like a birds belly?", "Answer": "a bottle"}}, "Round 3":{{"Question": "who said that?", "Answer": "Asta."}}, "Round 4":{{"Question": "Was Sharkie a friend??", "Answer": "Yes"}}, "Round 5":{{"Question": "did they get the bottle??", "Answer": "Yes"}}, "Round 6":{{"Question": "What was in it?", "Answer": "a note"}}, "Round 7":{{"Question": "Did a little boy write the note?", "Answer": "No"}}, "Round 8":{{"Question": "Who could read the note?", "Answer": "Asta\'s papa"}}, "Round 9":{{"Question": "What did they do with the note?", "Answer": "unknown"}}, "Round 10":{{"Question": "did they write back?", "Answer": "yes"}}, "Round 11":{{"Question": "were they excited?", "Answer": "unknown"}}}}
=====================================================================
Output:
{{
  "Round 1": {{
    "Question": "What was the name of the fish?",
    "Answer": "The name of the fish was Asta."
  }},
  "Round 2": {{
    "Question": "What looked like a bird's belly?",
    "Answer": "A bottle looked like a bird's belly."
  }},
  "Round 3": {{
    "Question": "Who said that the bottle looked like a bird's belly?",
    "Answer": "Asta said that the bottle looked like a bird's belly."
  }},
  "Round 4": {{
    "Question": "Was Sharkie a friend of Asta?",
    "Answer": "Yes, Sharkie was a friend of Asta."
  }},
  "Round 5": {{
    "Question": "Did Asta and Sharkie get the bottle?",
    "Answer": "Yes, Asta and Sharkie got the bottle."
  }},
  "Round 6": {{
    "Question": "What was in the bottle?",
    "Answer": "There was a note in the bottle."
  }},
  "Round 7": {{
    "Question": "Did a little boy write the note found in the bottle?",
    "Answer": "No, a little boy did not write the note found in the bottle."
  }},
  "Round 8": {{
    "Question": "Who could read the note found in the bottle?",
    "Answer": "Asta's papa could read the note found in the bottle."
  }},
  "Round 9": {{
    "Question": "What did Asta's papa do with the note?",
    "Answer": "What Asta's papa did with the note is unknown."
  }},
  "Round 10": {{
    "Question": "Did Asta's papa write back after reading the note?",
    "Answer": "Yes, Asta's papa wrote back after reading the note."
  }},
  "Round 11": {{
    "Question": "Were Asta and his friends excited after finding the note?",
    "Answer": "It is unknown if Asta and his friends were excited after finding the note."
  }}
}}
=====================================================================
Now generate your answer with real data:
Dialogue_history: {dialogue_content}
=====================================================================
Output:
"""
    # print(EXTRACT_FULL_INFO_QUESTION_PROMPT.format(dialogue_content=dialogue_content))
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": GENERAL_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": EXTRACT_FULL_INFO_QUESTION_PROMPT.format(dialogue_content=dialogue_content)
            }
        ],
        model="llama3-70b-8192",
        response_format={"type": "json_object"},
        temperature=llm_temperature,
        max_tokens=8192,
        top_p=1,
        stream=False,
        stop=None,
    )
    returned_content = chat_completion.choices[0].message.content

    if return_raw:
        return returned_content
    else:
        dict_round_information = eval(returned_content)
        return dict_round_information


def extract_topic(dialogue_content, return_raw=False):
    GENERATE_TOPIC_PROMPT="""
    You are an intelligent assistant that helps a human to analyze the information in a text document.
    Given a dialogue record, help the user by assigning a descriptive topic that summarizes what the dialogue is about.
    ====================================
    Text:{input_text}
    Output in JSON format: {{"topic":<topic_of_dialogue>}}
    Output:
    """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": GENERAL_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": GENERATE_TOPIC_PROMPT.format(input_text=dialogue_content)
            }
        ],
        model="llama3-8b-8192",
        response_format={"type": "json_object"},
        temperature=llm_temperature,
        max_tokens=8192,
        top_p=1,
        stream=False,
        stop=None,
    )
    returned_content = chat_completion.choices[0].message.content

    if return_raw:
        return returned_content
    else:
        return eval(returned_content)
    

def entity_types(input_topic, input_dialogue, llm_temperature=0):
    GRAPHRAG_GENERATE_ENTITY_TYPE_PROMPT="""
    The goal is to study the connections and relations between the entity types and their features in order to understand all available information from the text.

    The user's task is to investigate a dialogue with the topic of: {input_topic}

    As part of the analysis, you want to identify the entity types present in the following text.

    The entity types must be relevant to the user task.

    Avoid general entity types such as "other" or "unknown".

    This is VERY IMPORTANT: Do not generate redundant or overlapping entity types. For example, if the text contains "company" and "organization" entity types, you should return only one of them.

    Don't worry about quantity, always choose quality over quantity. And make sure EVERYTHING in your answer is relevant to the context of entity extraction.

    Return the entity types in JSON format with "entities_types" as the key and the entity types as an array of strings.

    =====================================================================

    EXAMPLE SECTION: The following section includes example output. These examples **must be excluded from your answer**.

    EXAMPLE 1
    Task: investigate Children's Literature.
    Text: What color was Cotton? white. Where did she live? in a barn. Did she live alone? no. Who did she live with? with her mommy and 5 sisters. What color were her sisters? orange and white. Was Cotton happy that she looked different than the rest of her family? no. What did she do to try to make herself the same color as her sisters? she painted herself. Whose paint was it? the farmer. What did Cotton's mother and siblings do when they saw her painted orange? they started laughing. Where did Cotton's mother put her to clean the paint off? a bucket of water. What did the other cats do when Cotton emerged from the bucket of water? licked her face. Did they want Cotton to change the color of her fur? no.
    JSON RESPONSE:
    {{"entity_types": [
        "character",
        "location",
        "object",
        "action",
        "emotion",
        "family"
    ] }}
    END OF EXAMPLE 1
    ======================================================================

    ======================================================================
    REAL DATA: The following section is the real data. You should use only this real data to prepare your answer. Generate Entity Types only.
    Task: investigate {input_topic}
    Text: {input_text}
    JSON response:
    {{"entity_types": [<entity_types>]}}
    Output:
    """

    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": GENERAL_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": GRAPHRAG_GENERATE_ENTITY_TYPE_PROMPT.format(input_topic=input_topic, input_text=input_dialogue)
            }
        ],
        temperature=llm_temperature,
        max_tokens=4096,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )
    entity_types = eval(completion.choices[0].message.content)["entity_types"]
    return entity_types


def entity_relations(entity_types, input_text, llm_temperature):
    GRAPH_EXTRACT_PROMPT = """
    -Goal-
    Given a document and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

    -Steps-
    1. Identify all entities. For each identified entity, extract the following information:
    - entity_name: Name of the entity
    - entity_type: One of the following types: {entity_types}
    - entity_description: Comprehensive description of the entity's attributes and activities
    Format each entity as JSON: {{"name":<entity_name>,"type":<entity_type>,"description":<entity_description>}}

    2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
    For each pair of related entities, extract the following information:
    - source_entity: name of the source entity, as identified in step 1
    - target_entity: name of the target entity, as identified in step 1
    - relationship_description: explanation as to why you think the source entity and the target entity are related to each other
    Format each relationship as JSON: {{"relationship":<relation_name>, "source_entity":<source_entity>, "target_entity": <target_entity>, "relationship_description":<relationship_description>}}

    3. Return output in JSON format with "entities" as the key to the array of strings of all the entities in steps 1, and "relations" as the key to the list of relationships identified in steps 2: {{"entities":[<entity_information_in_step_1>], "relations":[<relation_information_in_step_2>]}}


    =====================================================================
    Example:
    Entity_types: ["animal","location","person","object","action","emotion","family"]
    Text:
    "Cotton was white. Cotton lived in a barn. Cotton did not live alone. Cotton lived with her mommy and 5 sisters. Cotton's sisters were orange and white. Cotton was not happy that she looked different than the rest of her family. Cotton painted herself to try to make herself the same color as her sisters. The paint Cotton used belonged to the farmer. Cotton's mother and siblings started laughing when they saw her painted orange. Cotton's mother put her in a bucket of water to clean the paint off. The other cats licked Cotton's face when she emerged from the bucket of water. Cotton's family did not want her to change the color of her fur. "

    =====================================================================
    Output:
    {{"entities":[
    {{
        "name": "Cotton",
        "type": "character",
        "description": "A white kitten who lives in a barn with her mommy and five sisters. She feels unhappy because she looks different from her family and paints herself to resemble her sisters."
    }},
    {{
        "name": "Cotton's mommy",
        "type": "character",
        "description": "The mother of Cotton and her sisters, an orange and white cat who lives with them in the barn."
    }},
    {{
        "name": "Cotton's sisters",
        "type": "character",
        "description": "Five sisters of Cotton who are orange and white. They share the barn with Cotton and their mother."
    }},
    {{
        "name": "The farmer",
        "type": "character",
        "description": "The owner of the farm and the orange paint that Cotton uses to paint herself."
    }},
    {{
        "name": "Barn",
        "type": "location",
        "description": "The place where Cotton, her mommy, and sisters live."
    }},
    {{
        "name": "Paint",
        "type": "object",
        "description": "The orange paint belonging to the farmer that Cotton uses to paint herself."
    }},
    {{
        "name": "Bucket of water",
        "type": "object",
        "description": "The bucket of water where Cotton's mother places her to wash off the orange paint."
    }},
    {{
        "name": "Fur",
        "type": "object",
        "description": "Cotton's fur, which is white. She tries to change its color to match her sisters'."
    }},
    {{
        "name": "Cotton's face",
        "type": "object",
        "description": "Cotton's face, which her sisters lick to dry after she emerges from the bucket of water."
    }},
    {{
        "name": "Painted herself",
        "type": "action",
        "description": "Cotton paints herself with orange paint to try to look like her sisters."
    }},
    {{
        "name": "Started laughing",
        "type": "action",
        "description": "Cotton's mother and siblings laugh when they see her painted orange."
    }},
    {{
        "name": "Put her in a bucket of water",
        "type": "action",
        "description": "Cotton's mother puts her in a bucket of water to clean off the paint."
    }},
    {{
        "name": "Licked Cotton's face",
        "type": "action",
        "description": "Cotton's sisters lick her face to dry her after she emerges from the water."
    }},
    {{
        "name": "Unhappiness",
        "type": "emotion",
        "description": "Cotton feels unhappy because she looks different from the rest of her family."
    }},
    {{
        "name": "Family",
        "type": "family",
        "description": "Cotton's family, including her mommy and five sisters."
    }},
    {{
        "name": "White",
        "type": "color",
        "description": "The color of Cotton's fur."
    }},
    {{
        "name": "Orange and white",
        "type": "color",
        "description": "The colors of Cotton's sisters' fur."
    }},
    {{
        "name": "Orange",
        "type": "color",
        "description": "The color of the paint that Cotton uses to paint herself."
    }}],
    "relations":[
    {{
        "relationship": "mother-daughter",
        "source_entity": "Cotton",
        "target_entity": "Cotton's mommy",
        "relationship_description": "Cotton is the daughter of Cotton's mommy."
    }},
    {{
        "relationship": "siblings",
        "source_entity": "Cotton",
        "target_entity": "Cotton's sisters",
        "relationship_description": "Cotton and her sisters are siblings."
    }},
    {{
        "relationship": "lives in",
        "source_entity": "Cotton",
        "target_entity": "Barn",
        "relationship_description": "Cotton lives in the barn with her mommy and sisters."
    }},
    {{
        "relationship": "uses",
        "source_entity": "Cotton",
        "target_entity": "Paint",
        "relationship_description": "Cotton uses the farmer's orange paint to paint herself."
    }},
    {{
        "relationship": "owned by",
        "source_entity": "Paint",
        "target_entity": "The farmer",
        "relationship_description": "The orange paint belongs to the farmer."
    }},
    {{
        "relationship": "cleans with",
        "source_entity": "Cotton's mommy",
        "target_entity": "Bucket of water",
        "relationship_description": "Cotton's mommy cleans Cotton using the bucket of water."
    }},
    {{
        "relationship": "has",
        "source_entity": "Cotton",
        "target_entity": "Fur",
        "relationship_description": "Cotton has white fur."
    }},
    {{
        "relationship": "feels",
        "source_entity": "Cotton",
        "target_entity": "Unhappiness",
        "relationship_description": "Cotton feels unhappy because she looks different from her family."
    }},
    {{
        "relationship": "performs",
        "source_entity": "Cotton",
        "target_entity": "Painted herself",
        "relationship_description": "Cotton paints herself to try to look like her sisters."
    }},
    {{
        "relationship": "performs",
        "source_entity": "Cotton's sisters",
        "target_entity": "Licked Cotton's face",
        "relationship_description": "Cotton's sisters lick her face to dry her after she emerges from the water."
    }},
    {{
        "relationship": "performs",
        "source_entity": "Cotton's mother and siblings",
        "target_entity": "Started laughing",
        "relationship_description": "They laugh when they see Cotton painted orange."
    }},
    {{
        "relationship": "has color",
        "source_entity": "Cotton",
        "target_entity": "White",
        "relationship_description": "Cotton's fur is white."
    }},
    {{
        "relationship": "has color",
        "source_entity": "Cotton's sisters",
        "target_entity": "Orange and white",
        "relationship_description": "Cotton's sisters have orange and white fur."
    }},
    {{
        "relationship": "member of",
        "source_entity": "Cotton",
        "target_entity": "Family",
        "relationship_description": "Cotton is a member of her family."
    }},
    {{
        "relationship": "does not want to change",
        "source_entity": "Cotton's family",
        "target_entity": "Cotton",
        "relationship_description": "Cotton's family does not want her to change the color of her fur."
    }}
    ]}}
    END OF Example
    =====================================================================

    -Real Data-
    =====================================================================
    entity_types: {entity_types}
    text: {input_text}
    output:
    """

    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": GENERAL_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": GRAPH_EXTRACT_PROMPT.format(entity_types=entity_types, input_text=input_text)
            }
        ],
        temperature=llm_temperature,
        max_tokens=8192,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )
    list_entity_relations = eval(completion.choices[0].message.content)
    return list_entity_relations


def round_subgraph(entity_list, relation_list, dialogue_content, llm_temperature=0):
    ROUND_INVOLVED_INFO_PROMPT = """-Goal-
Given following information:
    entity_list: all entities, with their name, type and description for your reference
    relation_list: all relations, with their source_entity, target_entity, and relationship_description for your reference
    dialogue_record: multiple rounds question and answer from a dialogue

Read dialogue_record, select all entities involved in each question and answer from provided entity_list, select all relations involved in each question and answer from relation_list.

Output in JSON format: 
{{
"Round 1":{{
    "Question":{{
        "entities":[<entity name>, ...], 
        "relations":[{{"relationship":<relationship name>,"source_entity":<source entity name>, "target_entity":<target entity name>}}, ...]
    }},
    "Answer":{{
        "entities":[<entity name>, ...], 
        "relations":[{{"relationship":<relationship name>,"source_entity":<source entity name>, "target_entity":<target entity name>}}, ...]
    }},
}}, 
...}}

Example: The following section includes example output. These examples **must be excluded from your answer**.

=====================================================================
entity_list: [{{'name': 'Cotton',
  'type': 'character',
  'description': 'A white kitten who lives in a barn with her mommy and five sisters. She feels unhappy because she looks different from her family and paints herself to resemble her sisters.'}},
 {{'name': "Cotton's mommy",
  'type': 'character',
  'description': 'The mother of Cotton and her sisters, an orange and white cat who lives with them in the barn.'}},
 {{'name': "Cotton's sisters",
  'type': 'character',
  'description': 'Five sisters of Cotton who are orange and white. They share the barn with Cotton and their mother.'}},
 {{'name': 'The farmer',
  'type': 'character',
  'description': 'The owner of the farm and the orange paint that Cotton uses to paint herself.'}},
 {{'name': 'Barn',
  'type': 'location',
  'description': 'The place where Cotton, her mommy, and sisters live.'}},
 {{'name': 'Paint',
  'type': 'object',
  'description': 'The orange paint belonging to the farmer that Cotton uses to paint herself.'}},
 {{'name': 'Bucket of water',
  'type': 'object',
  'description': "The bucket of water where Cotton's mother places her to wash off the orange paint."}},
 {{'name': 'Fur',
  'type': 'object',
  'description': "Cotton's fur, which is white. She tries to change its color to match her sisters'."}},
 {{'name': "Cotton's face",
  'type': 'object',
  'description': "Cotton's face, which her sisters lick to dry after she emerges from the bucket of water."}},
 {{'name': 'Painted herself',
  'type': 'action',
  'description': 'Cotton paints herself with orange paint to try to look like her sisters.'}},
 {{'name': 'Started laughing',
  'type': 'action',
  'description': "Cotton's mother and siblings laugh when they see her painted orange."}},
 {{'name': 'Put her in a bucket of water',
  'type': 'action',
  'description': "Cotton's mother puts her in a bucket of water to clean off the paint."}},
 {{'name': "Licked Cotton's face",
  'type': 'action',
  'description': "Cotton's sisters lick her face to dry her after she emerges from the water."}},
 {{'name': 'Unhappiness',
  'type': 'emotion',
  'description': 'Cotton feels unhappy because she looks different from the rest of her family.'}},
 {{'name': 'Family',
  'type': 'family',
  'description': "Cotton's family, including her mommy and five sisters."}},
 {{'name': 'White',
  'type': 'color',
  'description': "The color of Cotton's fur."}},
 {{'name': 'Orange and white',
  'type': 'color',
  'description': "The colors of Cotton's sisters' fur."}},
 {{'name': 'Orange',
  'type': 'color',
  'description': 'The color of the paint that Cotton uses to paint herself.'}}]

relation_list:[{{'relationship': 'mother-daughter',
  'source_entity': 'Cotton',
  'target_entity': "Cotton's mommy",
  'relationship_description': "Cotton is the daughter of Cotton's mommy."}},
 {{'relationship': 'siblings',
  'source_entity': 'Cotton',
  'target_entity': "Cotton's sisters",
  'relationship_description': 'Cotton and her sisters are siblings.'}},
 {{'relationship': 'lives in',
  'source_entity': 'Cotton',
  'target_entity': 'Barn',
  'relationship_description': 'Cotton lives in the barn with her mommy and sisters.'}},
 {{'relationship': 'uses',
  'source_entity': 'Cotton',
  'target_entity': 'Paint',
  'relationship_description': "Cotton uses the farmer's orange paint to paint herself."}},
 {{'relationship': 'owned by',
  'source_entity': 'Paint',
  'target_entity': 'The farmer',
  'relationship_description': 'The orange paint belongs to the farmer.'}},
 {{'relationship': 'cleans with',
  'source_entity': "Cotton's mommy",
  'target_entity': 'Bucket of water',
  'relationship_description': "Cotton's mommy cleans Cotton using the bucket of water."}},
 {{'relationship': 'has',
  'source_entity': 'Cotton',
  'target_entity': 'Fur',
  'relationship_description': 'Cotton has white fur.'}},
 {{'relationship': 'feels',
  'source_entity': 'Cotton',
  'target_entity': 'Unhappiness',
  'relationship_description': 'Cotton feels unhappy because she looks different from her family.'}},
 {{'relationship': 'performs',
  'source_entity': 'Cotton',
  'target_entity': 'Painted herself',
  'relationship_description': 'Cotton paints herself to try to look like her sisters.'}},
 {{'relationship': 'performs',
  'source_entity': "Cotton's sisters",
  'target_entity': "Licked Cotton's face",
  'relationship_description': "Cotton's sisters lick her face to dry her after she emerges from the water."}},
 {{'relationship': 'performs',
  'source_entity': "Cotton's mother and siblings",
  'target_entity': 'Started laughing',
  'relationship_description': 'They laugh when they see Cotton painted orange.'}},
 {{'relationship': 'has color',
  'source_entity': 'Cotton',
  'target_entity': 'White',
  'relationship_description': "Cotton's fur is white."}},
 {{'relationship': 'has color',
  'source_entity': "Cotton's sisters",
  'target_entity': 'Orange and white',
  'relationship_description': "Cotton's sisters have orange and white fur."}},
 {{'relationship': 'member of',
  'source_entity': 'Cotton',
  'target_entity': 'Family',
  'relationship_description': 'Cotton is a member of her family.'}},
 {{'relationship': 'does not want to change',
  'source_entity': "Cotton's family",
  'target_entity': 'Cotton',
  'relationship_description': "Cotton's family does not want her to change the color of her fur."}}]

dialogue_record: 
{{
    "Round 1": {{
        "Question": "What color was Cotton?",
        "Answer": "Cotton was white."
    }},
    "Round 2": {{
        "Question": "Where did Cotton live?",
        "Answer": "Cotton lived in a barn."
    }},
    "Round 3": {{
        "Question": "Did Cotton live alone in the barn?",
        "Answer": "No, Cotton did not live alone in the barn."
    }},
    "Round 4": {{
        "Question": "Who did Cotton live with in the barn?",
        "Answer": "Cotton lived with her mommy and 5 sisters in the barn."
    }},
    "Round 5": {{
        "Question": "What color were Cotton's sisters?",
        "Answer": "Cotton's sisters were orange and white."
    }},
    "Round 6": {{
        "Question": "Was Cotton happy that she looked different than her mommy and sisters?",
        "Answer": "No, Cotton was not happy that she looked different than her mommy and sisters."
    }},
    "Round 7": {{
        "Question": "What did Cotton do to try to make herself the same color as her sisters?",
        "Answer": "Cotton painted herself to try to make herself the same color as her sisters."
    }},
    "Round 8": {{
        "Question": "Whose paint did Cotton use to paint herself?",
        "Answer": "Cotton used the farmer's paint to paint herself."
    }},
    "Round 9": {{
        "Question": "What did Cotton's mother and siblings do when they saw her painted orange?",
        "Answer": "Cotton's mother and siblings started laughing when they saw her painted orange."
    }},
    "Round 10": {{
        "Question": "Where did Cotton's mother put her to clean the paint off?",
        "Answer": "Cotton's mother put her in a bucket of water to clean the paint off."
    }},
    "Round 11": {{
        "Question": "What did the other cats do when Cotton emerged from the bucket of water?",
        "Answer": "The other cats licked Cotton's face when she emerged from the bucket of water."
    }},
    "Round 12": {{
        "Question": "Did Cotton's family want her to change the color of her fur?",
        "Answer": "No, Cotton's family did not want her to change the color of her fur."
    }}
}}

Output:
{{
  "Round 1": {{
    "Question": {{
      "entities": ["Cotton", "White"],
      "relations": [
        {{
          "relationship": "has color",
          "source_entity": "Cotton",
          "target_entity": "White"
        }}
      ]
    }},
    "Answer": {{
      "entities": ["Cotton", "White"],
      "relations": [
        {{
          "relationship": "has color",
          "source_entity": "Cotton",
          "target_entity": "White"
        }}
      ]
    }}
  }},
  "Round 2": {{
    "Question": {{
      "entities": ["Cotton", "Barn"],
      "relations": [
        {{
          "relationship": "lives in",
          "source_entity": "Cotton",
          "target_entity": "Barn"
        }}
      ]
    }},
    "Answer": {{
      "entities": ["Cotton", "Barn"],
      "relations": [
        {{
          "relationship": "lives in",
          "source_entity": "Cotton",
          "target_entity": "Barn"
        }}
      ]
    }}
  }},
  "Round 3": {{
    "Question": {{
      "entities": ["Cotton", "Barn"],
      "relations": [
        {{
          "relationship": "lives in",
          "source_entity": "Cotton",
          "target_entity": "Barn"
        }}
      ]
    }},
    "Answer": {{
      "entities": ["Cotton", "Barn", "Cotton's mommy", "Cotton's sisters"],
      "relations": [
        {{
          "relationship": "lives in",
          "source_entity": "Cotton",
          "target_entity": "Barn"
        }}
      ]
    }}
  }},
  "Round 4": {{
    "Question": {{
      "entities": ["Cotton", "Barn", "Cotton's mommy", "Cotton's sisters"],
      "relations": [
        {{
          "relationship": "lives in",
          "source_entity": "Cotton",
          "target_entity": "Barn"
        }}
      ]
    }},
    "Answer": {{
      "entities": ["Cotton", "Cotton's mommy", "Cotton's sisters", "Barn"],
      "relations": [
        {{
          "relationship": "lives in",
          "source_entity": "Cotton",
          "target_entity": "Barn"
        }}
      ]
    }}
  }},
  "Round 5": {{
    "Question": {{
      "entities": ["Cotton's sisters", "Orange and white"],
      "relations": [
        {{
          "relationship": "has color",
          "source_entity": "Cotton's sisters",
          "target_entity": "Orange and white"
        }}
      ]
    }},
    "Answer": {{
      "entities": ["Cotton's sisters", "Orange and white"],
      "relations": [
        {{
          "relationship": "has color",
          "source_entity": "Cotton's sisters",
          "target_entity": "Orange and white"
        }}
      ]
    }}
  }},
  "Round 6": {{
    "Question": {{
      "entities": ["Cotton", "Unhappiness", "Cotton's mommy", "Cotton's sisters"],
      "relations": [
        {{
          "relationship": "feels",
          "source_entity": "Cotton",
          "target_entity": "Unhappiness"
        }}
      ]
    }},
    "Answer": {{
      "entities": ["Cotton", "Unhappiness", "Cotton's mommy", "Cotton's sisters"],
      "relations": [
        {{
          "relationship": "feels",
          "source_entity": "Cotton",
          "target_entity": "Unhappiness"
        }}
      ]
    }}
  }},
  "Round 7": {{
    "Question": {{
      "entities": ["Cotton", "Painted herself", "Cotton's sisters"],
      "relations": [
        {{
          "relationship": "performs",
          "source_entity": "Cotton",
          "target_entity": "Painted herself"
        }}
      ]
    }},
    "Answer": {{
      "entities": ["Cotton", "Painted herself", "Cotton's sisters"],
      "relations": [
        {{
          "relationship": "performs",
          "source_entity": "Cotton",
          "target_entity": "Painted herself"
        }}
      ]
    }}
  }},
  "Round 8": {{
    "Question": {{
      "entities": ["Cotton", "Paint", "The farmer"],
      "relations": [
        {{
          "relationship": "uses",
          "source_entity": "Cotton",
          "target_entity": "Paint"
        }},
        {{
          "relationship": "owned by",
          "source_entity": "Paint",
          "target_entity": "The farmer"
        }}
      ]
    }},
    "Answer": {{
      "entities": ["Cotton", "Paint", "The farmer"],
      "relations": [
        {{
          "relationship": "uses",
          "source_entity": "Cotton",
          "target_entity": "Paint"
        }},
        {{
          "relationship": "owned by",
          "source_entity": "Paint",
          "target_entity": "The farmer"
        }}
      ]
    }}
  }},
  "Round 9": {{
    "Question": {{
      "entities": ["Cotton's mother and siblings", "Started laughing", "Cotton"],
      "relations": [
        {{
          "relationship": "performs",
          "source_entity": "Cotton's mother and siblings",
          "target_entity": "Started laughing"
        }}
      ]
    }},
    "Answer": {{
      "entities": ["Cotton's mother and siblings", "Started laughing", "Cotton"],
      "relations": [
        {{
          "relationship": "performs",
          "source_entity": "Cotton's mother and siblings",
          "target_entity": "Started laughing"
        }}
      ]
    }}
  }},
  "Round 10": {{
    "Question": {{
      "entities": ["Cotton's mommy", "Bucket of water"],
      "relations": [
        {{
          "relationship": "cleans with",
          "source_entity": "Cotton's mommy",
          "target_entity": "Bucket of water"
        }}
      ]
    }},
    "Answer": {{
      "entities": ["Cotton's mommy", "Bucket of water"],
      "relations": [
        {{
          "relationship": "cleans with",
          "source_entity": "Cotton's mommy",
          "target_entity": "Bucket of water"
        }}
      ]
    }}
  }},
  "Round 11": {{
    "Question": {{
      "entities": ["Cotton", "Bucket of water", "Cotton's sisters", "Licked Cotton's face"],
      "relations": [
        {{
          "relationship": "performs",
          "source_entity": "Cotton's sisters",
          "target_entity": "Licked Cotton's face"
        }}
      ]
    }},
    "Answer": {{
      "entities": ["Cotton", "Bucket of water", "Cotton's sisters", "Licked Cotton's face"],
      "relations": [
        {{
          "relationship": "performs",
          "source_entity": "Cotton's sisters",
          "target_entity": "Licked Cotton's face"
        }}
      ]
    }}
  }},
  "Round 12": {{
    "Question": {{
      "entities": ["Cotton's family", "Cotton", "Fur"],
      "relations": [
        {{
          "relationship": "does not want to change",
          "source_entity": "Cotton's family",
          "target_entity": "Cotton"
        }}
      ]
    }},
    "Answer": {{
      "entities": ["Cotton's family", "Cotton", "Fur"],
      "relations": [
        {{
          "relationship": "does not want to change",
          "source_entity": "Cotton's family",
          "target_entity": "Cotton"
        }}
      ]
    }}
  }}
}}

END OF EXAMPLE
=====================================================================
REAL DATA: The following section is the real data. You should use only this real data to prepare your answer.

entity_list: {entity_list}

relation_list: {relation_list}  

dialogue_record: {dialogue_content}

Output:{{
"Round 1":{{
    "Question":{{
        "entities":[<entity name>, ...], 
        "relations":[{{"relationship":<relationship name>,"source_entity":<source entity name>, "target_entity":<target entity name>}}, ...]
    }},
    "Answer":{{
        "entities":[<entity name>, ...], 
        "relations":[{{"relationship":<relationship name>,"source_entity":<source entity name>, "target_entity":<target entity name>}}, ...]
    }},
}}, 
...}}
"""

    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": GENERAL_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": ROUND_INVOLVED_INFO_PROMPT.format(entity_list = entity_list, relation_list = relation_list, dialogue_content=dialogue_content)
            }
        ],
        temperature=llm_temperature,
        max_tokens=8192,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )
    result = eval(completion.choices[0].message.content)
    return result


def entity_subset_filter(entity_list, entity_types, target_entity_name, llm_temperature=0):
    pure_names = [item["name"] for item in entity_list]
    # print("using LLM for: ",target_entity_name, "entity_types:",entity_types)
    ENTITY_SUBSET_PROMPT="""-Goal-
Given the following information:
existing entity_list,
entity_types,
name_of_the_new_entity

Task: 
Determine whether the new entity:name_of_the_new_entity, is a subset of existing entities in entity_list, the subset can be a group of one or more entities from entity_list. 
If yes, fill the "is_subset_flag" field with 1, choose corresponding entities from entity_list, and put their names and types in the "members" field. 
If no, fill the "is_subset_flag" field with 0, choose the most appropriate entity type from the entity_types, and fill the new entity's name and type into the "members" field.

Output in JSON format:
{{"is_subset_flag": <0 or 1>,
"members":[{{"name":<name_of_entity>, "type": <type_of_entity>}},]
}}
=====================================================================
Example: The following section includes example output. These examples **must be excluded from your answer**.

existing entity_list: [{{'name': 'Cotton', 'type': 'character', 'description': 'A white kitten who lives in a barn with her mommy and five sisters. She feels unhappy because she looks different from her family and paints herself to resemble her sisters.'}}, {{'name': "Cotton's mommy", 'type': 'character', 'description': 'The mother of Cotton and her sisters, an orange and white cat who lives with them in the barn.'}}, {{'name': "Cotton's sisters", 'type': 'character', 'description': 'Five sisters of Cotton who are orange and white. They share the barn with Cotton and their mother.'}}, {{'name': 'The farmer', 'type': 'character', 'description': 'The owner of the farm and the orange paint that Cotton uses to paint herself.'}}]

entity_types: ['character', 'location', 'object']

name_of_the_new_entity: Cotton's mother and siblings

Output:{{"is_subset_flag": 1,"members": [{{"name": "Cotton's mommy","type": "character"}},{{"name": "Cotton's sisters","type": "character"}}]}}

END OF EXAMPLE
=====================================================================
REAL DATA: The following section is the real data. You should use only this real data to prepare your answer.

existing entity_list: {entity_list}
entity_types: {entity_types}
name of a new entity:  {target_entity_name}

Output:
"""
    completion = client.chat.completions.create(
        model="llama3-70b-8192",

        messages=[
            {
                "role": "user",
                "content": ENTITY_SUBSET_PROMPT.format(entity_list=entity_list, entity_types=entity_types, target_entity_name=target_entity_name)
            }
        ],
        temperature=llm_temperature,
        max_tokens=4096,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )
    result = eval(completion.choices[0].message.content)
    # print(result)
    if "is_subset_flag" in result.keys():
        if "members" in result.keys():
            if False not in [("name" in member.keys() or "type" in member.keys()) for member in result["members"]]:
                if (result["is_subset_flag"]==1 and False not in [member["name"] in pure_names for member in result["members"]]):
                    return result
                elif (result["is_subset_flag"]==0 and result["members"][0]["type"] in entity_types):
                    return result
                    

    raise(SyntaxError("LLM generated bad result: \ninput: {} \n=>\nOutput{}".format((entity_list, entity_types, target_entity_name),result)))
    
