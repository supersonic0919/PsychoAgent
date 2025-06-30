import sys
import io
import json
import pandas as pd
import requests
import re
from openai import OpenAI
from tqdm import tqdm
import time
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from concurrent.futures import ThreadPoolExecutor

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com "
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') 

Excel_PATH = r"datafiles\\hurricane_weather_data_1.xlsx"
# File path configuration
# USER_PROFILE_PATH = r"datafiles\\COPE\\merged_user_files.csv"
# Validation set file consistent with baseline method (20%)
USER_PROFILE_PATH = r"datafiles\\COPE\\merged_user_files_1.csv"
TWEET_DATA_PATH = r"datafiles\\COPE\\cleaned_Formed_data.csv"
OUTPUT_PATH = r"outputFiles\\CoT_rolePlaying_1.csv"

# Global loading of psychological knowledge
psychology = """
    1. Public Risk Perception Formation:
   - Risk perception is shaped by two factors and their interaction: 
     a) Characteristics of the risk event itself
     b) Personal characteristics of the audience


    2. Personality Traits and Risk Response:
    Psychoticism: Users with higher psychoticism may overestimate their ability to control events, potentially leading to riskier behaviors.
    Neuroticism: Neuroticism affects emergency comprehension and fear levels. Users with neuroticism above 0.537 may experience higher fear and prefer passive coping strategies, whereas those below this threshold may remain calmer in crises.
    Extraversion: Extraversion is linked to perceived understanding of emergencies. Users with extraversion above 0.525 tend to adopt proactive measures, while those below this threshold may be more reserved in their response.
    Agreeableness: Agreeableness influences cooperative behavior during risks. Users with agreeableness above 0.449 may seek harmony and follow recommended actions, while others might question authorities.
    Conscientiousness: Conscientiousness relates to organized responses. Users with conscientiousness above 0.304 may adhere strictly to safety protocols, whereas those below may improvise.
    Openness: Openness affects adaptability. Users with openness above 0.522 may explore innovative solutions, while others might rely on conventional approaches.
    
    3. Social Media Language Style Effects:
    - Sarcasm/irony may amplify anxiety in crisis contexts

    4. Content Type Emotional Impacts:
    - Disaster-related serious news increases situational awareness but may elevate stress

    5. Emotional Stability Mechanisms:
    - Regular use of cognitive reappraisal strategies buffers acute stress during disasters
   
    6. Social Media Network Characteristics and Panic Formation:
    - Users with more follows/followers are more likely to be exposed to diverse and potentially conflicting information, which can increase cognitive load and anxiety
    - Dense social networks (many friends) can lead to group polarization and echo chamber effects, amplifying panic through frequent interactions
    - Social comparison on platforms with many users can weaken self-efficacy when others display superior coping resources
    """

'''
    - High Psychoticism: Associated with overestimation of event controllability
    - High Extraversion: Correlates with perceived understanding of emergencies (e.g. pandemic knowledge)
    - High Neuroticism: Linked to lower emergency comprehension and higher fear levels
    - Extraverts tend to adopt proactive measures
    - Emotionally unstable individuals (high Neuroticism) prefer passive coping strategies

'''

# Contriever model configuration
CONTRIEVER_MODEL_PATH = "modelFiles\\contriever"
# Disaster event description (including core events and related topics)
HURRICANE_DESCRIPTION = {
    "core": "Hurricane Sandy, tropical storm, extreme weather",
    "response": "rescue operations, emergency response, disaster relief",
    "impact": "government actions, evacuation procedures, infrastructure damage",
    "preparation": "supply distribution, supply preparation, power restoration, medical aid",
    "secondary": "flooding, power outages, transportation disruption"
}
MAX_INPUT_LENGTH = 512

def load_data(n_users=1, start_row =0):
    """Load user features and tweet data"""
    try:
        # Load user features
        user_df = pd.read_csv(USER_PROFILE_PATH, 
                              skiprows=lambda i: i > 0 and i <= start_row,  
                              nrows=n_users)
        
        # Load historical tweet data
        tweet_df = pd.read_csv(TWEET_DATA_PATH, usecols=['user_id', 'text'])
        
        # Merge tweet data
        tweet_group = tweet_df.groupby('user_id')['text'].apply(list).reset_index()
        merged_df = pd.merge(user_df, tweet_group, on='user_id', how='left')
        
        return merged_df
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        sys.exit(1)

def parse_list(value):
    """Parse list-type data"""
    try:
        if pd.isna(value):
            return []
        return eval(str(value))
    except:
        return []

def sentiment_stability(sentiment_trend):
    """Calculate standard deviation of sentiment trend"""
    # sentiment_trend = row['sentiment_trend']
    if isinstance(sentiment_trend, list) and len(sentiment_trend) > 0:
        sentiment_values = [float(x) for x in sentiment_trend]  
        std_dev = np.std(sentiment_values)
    else:
        std_dev = 0.0  

    # Provide explanation based on standard deviation
    if std_dev < 0.3:
        emotional_stability = "The emotional trend is very stable with minimal fluctuations."
    elif 0.3 <= std_dev <= 0.5:
        emotional_stability = "The emotional trend has some fluctuations."
    else:
        emotional_stability = "The emotional trend has significant fluctuations and is not stable."
    return emotional_stability

# Initialize Contriever model
def load_contriever_model():
    """Load retrieval model and tokenizer"""
    try:
        # print("Loading Contriever model...")
        tokenizer = AutoTokenizer.from_pretrained(CONTRIEVER_MODEL_PATH)
        model = AutoModel.from_pretrained(CONTRIEVER_MODEL_PATH)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # print(f"Model loaded on {device}")
        
        return tokenizer, model
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        sys.exit(1)

# Initialize global model
tokenizer, model = load_contriever_model()

def mean_pooling(token_embeddings, mask):
    """Mean pooling function"""
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]

def get_relevant_tweets(text_list, top_k=5):
    """
    Get user's tweets most relevant to hurricanes
    :param text_list: List of tweet data
    :param top_k: Number of tweets to return
    :return: List of relevant tweets
    """
    if len(text_list) == 0:
        return []
    
    # Structured multidimensional query (topic + keywords)
    query_template = [
        ("Natural disaster event", HURRICANE_DESCRIPTION["core"]),
        ("Emergency response actions", HURRICANE_DESCRIPTION["response"]),
        ("Social impact aspects", HURRICANE_DESCRIPTION["impact"]),
        ("Preparation measures", HURRICANE_DESCRIPTION["preparation"]),
        ("Secondary disasters", HURRICANE_DESCRIPTION["secondary"])
    ]
    
    scores = []
    tweets = []
    for tweet in text_list:
        tweets.append(tweet)  
        tweet_scores = []
        for (context, keywords) in query_template:
            query = f"{context} related to {keywords}"
            inputs = tokenizer(
                [query, tweet],
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_LENGTH,
                return_tensors='pt'
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            score = F.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
            tweet_scores.append(score)
        
        scores.append(max(tweet_scores))
    
    # Get top_k tweets (maintain original order)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [tweets[i] for i in sorted_indices]

def build_user_profile(row):
    """Build user profile dictionary"""
    return {
        "user_id": row['user_id'],
        "location": [row['lat_final'], row['lng_final']],
        "big_five": {  
            'E': row['Extroversion'],
            'N': row['Neuroticism'],
            'A': row['Agreeableness'],
            'C': row['Conscientiousness'],
            'O': row['Openness']
        },
        "twitter_stats": {
            "followers": int(row['user_followers_count']),
            "following": int(row['user_friends_count']),
            "daily_tweets": float(row['text_average_count'])
        },
        "tone_of_voice": row['tone_of_voice'].split(', ') if pd.notna(row['tone_of_voice']) else [],
        "emotional_stability": sentiment_stability(parse_list(row['sentiment_trend'])),  
        "interests": row['topic'].split(', ') if pd.notna(row['topic']) else [],
        "recent_tweets": get_relevant_tweets(row['text'])
    }


def load_hurricane_data():
    """Load hurricane data from CSV and format as Markdown table"""
    try:  
        df = pd.read_excel(Excel_PATH)  
        
        df = df.rename(columns={
            'wind': 'Wind (km/h)',
            'air_pressure': 'Air Pressure (hPa)',
            'category': 'Category'
        })
        
        markdown_table = "| " + " | ".join(df.columns) + " |\n"  
        markdown_table += "|" + "|".join(["---"]*len(df.columns)) + "|\n"  
        for i, row in df.iterrows():  
            markdown_table += "| " + " | ".join(map(str, row.values)) + " |\n"  
            
        # Add impact scale table
        markdown_table += "\nImpact Scale Reference:\n"
        markdown_table += "| Category | Wind Range (km/h) | Storm Surge | Typical Damage Characteristics |\n"
        markdown_table += "|----------|-------------------|-------------|--------------------------------|\n"
        markdown_table += "| D | ≤62 | <0.3m | Road flooding, branch breakage |\n"
        markdown_table += "| S | 63-118 | 0.3-1.2m | Window breakage, temporary structure collapse |\n"
        markdown_table += "| H1 | 119-153 | 1.2-1.8m | Partial roof damage, prolonged power outages, multi-day power outages|\n"
        markdown_table += "| H2 | 154-177 | 1.8-2.4m | Structural damage, widespread tree falls |\n"
        markdown_table += "| H3 | 178-208 | 2.4-3.7m | Building frame exposure, complete roof failure, coastal flooding |\n"
        markdown_table += "| E | Variable | Tide-dependent | Combined wind-rain-snow disasters |\n"
        return markdown_table  
    except Exception as e:  
        print(f"Excel loading error: {str(e)}")  
        exit(1)
        

def build_questions():
    """Build complete list of 18 questions"""
    return [
        "I am familiar with the natural hazard/disaster preparedness materials relevant to my area",
        "I know how to adequately prepare my home for the forthcoming fire/flood/cyclone season",
        "I know which household preparedness measures are needed to stay safe in a natural hazard/disaster",
        "I know what to look out for in my home and workplace if an emergency weather situation should develop",
        "I am familiar with the disaster warning system messages used for extreme weather events",
        "I am confident that I know what to do and what actions to take in a severe weather situation",
        "I would be able to locate the natural hazard/disaster preparedness materials in a warning situation easily",
        "I am knowledgeable about the impact that a natural hazard/disaster can have on my home",
        "I know what the difference is between a disaster warning and a disaster watch situation",
        "I am familiar with the weather signs of an approaching fire/flood/cyclone",
        "I think I am able to manage my feelings pretty well in difficult and challenging situations",
        "In a natural hazard/disaster situation I would be able to cope with my anxiety and fear",
        "I seem to be able to stay cool and calm in most difficult situations",
        "I feel reasonably confident in my own ability to deal with stressful situations that I might find myself in",
        "When necessary, I can talk myself through challenging situations",
        "If I found myself in a natural hazard/disaster situation I would know how to manage my own response to the situation",
        "I know which strategies I could use to calm myself in a natural hazard/disaster situation",
        "I have a good idea of how I would likely respond in an emergency situation"
    ]

 
def parse_answers(response_text):
    """Parse answers to 18 questions, return format with question, answer, and reason"""
    answers = {}
    pattern = r"Q(\d+)\s*:\s*(\d+)\s*\(([^)]+?)\)"
    matches = re.findall(pattern, response_text, re.MULTILINE)
    
    if not matches:
        print("No matches found! Original response:\n", response_text)
    
    questions = build_questions()  
    for match in matches:
        q_num = int(match[0])
        q_text = questions[q_num - 1] if q_num - 1 < len(questions) else f"Question {q_num}"
        answers[f"Q{q_num}"] = {
            "score": int(match[1]),
            "reason": match[2].strip(),
            "text": q_text  
        }
    
    # Verify completeness
    if len(answers) != 18:
        raise ValueError(f"Parsed {len(answers)} answers, expected 18")
        
    return answers

def build_emotional_arousal_knowledge():
    """Build prior knowledge about emotional arousal"""
    return """
    Panic Emotion Arousal Factors in Risk Perception:

    1. Awareness of Danger:
    - The more aware people are of a risk, the more likely they are to worry about it.

    2. Coping Efficacy and Sense of Control:
    - The stronger people feel their control over a situation, the less panicked they tend to be.

    3. Uncertainty of Risk:
    - The more uncertain people are, the more likely they are to feel afraid and protect themselves with vigilance and fear.
    - Forms of uncertainty include:
        a. "I cannot detect it": Risks that are invisible or imperceptible.
        b. "I do not understand it": Lack of understanding of the scientific explanation of the risk.
        c. "No one knows": New risks where even scientists are uncertain about the nature, principles, and mechanisms.
    4. Novelty of Risk:
    - New risks often trigger more fear and attention.
    - As people become more familiar with the risk, fear decreases
    """
    
def construct_conversation(phase, user_info, answers=None, emotional_arousal_summary=None, failure_reason=None, tweet=None):
    """Construct complete conversation context"""
    messages = []  
    
    # Phase 1: Psychological knowledge
    # Changed to global variable, loaded directly in message
    
    # Phase 2: Hurricane data
    hurricane_table = load_hurricane_data()
    
    user_msg = f"""Use these resources:
    1. Psychological Principles:
    {psychology}

    2. Hurricane monitoring data (Markdown):
    {hurricane_table}

    3. User Profile (JSON):
    {json.dumps(user_info, indent=2)}
    
    Key Threshold Checks (scaled by 1000x):
    - Neuroticism: {user_info['big_five']['N']*1000:.2f} ({'ABOVE' if user_info['big_five']['N']*1000 > 537.37 else 'BELOW'} population average)
    - Extraversion: {user_info['big_five']['E']*1000:.2f} ({'ABOVE' if user_info['big_five']['E']*1000 > 525.003 else 'BELOW'} threshold)
    - Agreeableness: {user_info['big_five']['A']*1000:.2f} ({'ABOVE' if user_info['big_five']['A']*1000 > 449.303 else 'BELOW'} population mean)
    - Conscientiousness: {user_info['big_five']['C']*1000:.2f} ({'ABOVE' if user_info['big_five']['C']*1000 > 304.057 else 'BELOW'} baseline)
    - Openness: {user_info['big_five']['O']*1000:.2f} ({'ABOVE' if user_info['big_five']['O']*1000 > 522.084 else 'BELOW'} norm)
    
    Please always:\n1. Directly output the final answer\n2. Disable any thought process\n3. Use plain text format
    """
    messages.extend([
        {"role": "system", "content": "You are a psychologist specializing in predicting public emotional trends during emergencies."},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": "Data understood."}
    ])
    
    # Phase 4: Question answering
    if phase == "assessment":
        questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(build_questions())])
        messages.append(
            {"role": "user", 
            "content": f"""Answer the following questions sequentially (show each answer separately).For each question below, follow this reasoning chain:
    1. Question Comprehension: Analyze what specific knowledge/attitude the question assesses
    2. Memory Retrieval: Search long-term memory for relevant features from the following sources:
        -Psychological prior knowledge 
        -Real-world hurricane data (e.g., intensity, location, historical impact)
        -User-specific characteristics (e.g., personality traits, behavioral patterns, expressed attitudes)
    3. Option Mapping: For each rating (1-4):
        a) Reason about why the Participant might answer with that particular option.
        b) Identify memory evidence supporting/contradicting
    4. Synthesis: Weigh evidence to predict most likely response
    Questions:
    {questions}

    Response Requirements:
    1. Format: 'Q[number]: [score] (reason)'
    2. Use 1-4 rating scale:
        1: Completely Disagree (0-25% alignment)
        2: Somewhat Agree (26-50% alignment)
        3: Mostly Agree (51-75% alignment)
        4: Completely Agree (76-100% alignment)
    3. Follow these rules:
        a) Maintain consistency with psychological principles
        b) Consider the hurricane disaster context
        c) Fully embody the user's characteristics and answer strictly from the user's perspective
    4. Do NOT use Markdown formatting
    5. Number all answers 
    6. Do NOT include any internal thinking process or reasoning
    7. Provide answers in direct format without any additional text
    """
        })
    
    # Phase 5: Emotional arousal
    elif phase == "emotional_arousal":
        # Build risk assessment summary with questions, answers and reasons
        risk_assessment = []
        for q, ans in answers.items():
            risk_assessment.append(
                f"{q}: I {'completely' if ans['score'] == 4 else ('mostly' if ans['score'] == 3 else ('somewhat' if ans['score'] == 2 else 'completely disagree'))} agree that {ans['text']}, because {ans['reason'][:500]}..."
            )
        risk_assessment_summary = "\n".join(risk_assessment)
        # Build emotional arousal knowledge
        emotional_arousal_knowledge = build_emotional_arousal_knowledge()
        messages.extend([
                {
                    "role": "user",
                    "content": f"""User risk assessment result:\n{risk_assessment_summary}"""
                },
                {
                    "role": "assistant", 
                    "content": "Risk assessment integrated."
                },
                {
                    "role": "user",
                    "content": f"""Learn the following knowledge about panic emotion arousal factors:
                    {emotional_arousal_knowledge}"""
                },
                {
                    "role": "assistant", 
                    "content": "Panic emotion arousal factors understood."
                },
                {
                    "role": "user",
                    "content": f"""Based on the previous stages (psychological principles, hurricane data, user profile, and assessment answers), silently analyze these panic factors and assign a score (1-5) to each (DO NOT OUTPUT ANALYSIS):
    1. Awareness of Danger: Describe whether the user is aware of the danger.
    2. Coping Efficacy and Sense of Control: Describe the user's perceived control over the situation, comprehensively consider both their sentiment trends and personality dimensions.
    3. Uncertainty of Risk: Describe the user's understanding and ability to detect the risk
    4. Novelty of Risk: Describe the user's familiarity with the risk
    For each factor, provide:
    - A score from 1 to 5 (1: lowest, 5: highest)
    - A brief reason
    
    Scoring Rubric:
        Factor | Low (1-2) | Medium-High (3) | High (4-5)
        --- | --- | --- | ---
        Awareness | Minimal understanding of hurricane dangers (e.g., unaware of specific risks like flooding). | Basic risk recognition(e.g., aware of general risks) | deep awareness of hurricane dangers(e.g., understands specific risks like flooding, wind damage, etc.)
        Coping | Low self-efficacy(e.g., no confidence in handling crises). | Moderate confidence(e.g., some belief in ability to manage situations). | Strong crisis management (e.g., high confidence and clear strategies)
        Uncertainty | Familiar mechanisms(e.g., understands risks well). | Some unknowns(e.g., partial understanding of risks). | Complete confusion(e.g., no understanding of risks or how to respond).
        Novelty | familiarity with hurricane experience(e.g., extensive prior experience).| Seen similar events before(e.g., limited prior exposure)  | First exposure to hurricane-like events(e.g., no prior experience)
    Additional Guidance:
    - Low (1-2): Represents minimal understanding, confidence, or experience, with 2 indicating a slightly higher level than 1 but still below the medium threshold.
    - Medium (3): Neutral state, indicating balanced awareness, confidence, or experience.
    - High (4-5): Represents advanced understanding, confidence, or experience, with 4 indicating a slightly lower level than 5 but above the medium threshold.
    
    Calculation Rules:
    - Each factor contributes 25% weight to the panic probability.
    - The base probability is 50% when all factors are scored 3.
    - For each factor:
        - **Awareness**: Higher scores increase panic probability by 5% per point above 3.
        - **Coping**: Lower scores increase panic probability by 5% per point below 3.
        - **Uncertainty**: Higher scores increase panic probability by 5% per point above 3.
        - **Novelty**: Higher scores increase panic probability by 5% per point above 3.
    Example:
    - If Awareness is 4, Coping is 3, Uncertainty is 3, and Novelty is 3: Panic probability = 50 + (4-3)*5 + (3-3)*5 + (3-3)*5 + (3-3)*5 = 55%.
    - If Awareness is 5, Coping is 3, Uncertainty is 3, and Novelty is 3: Panic probability = 50 + (5-3)*5 + (3-3)*5 + (3-3)*5 + (3-3)*5 = 60%.
    
    Output format:
        Awareness: [1-5]/5 (reason)
        Coping: [1-5]/5 (reason)
        Uncertainty: [1-5]/5 (reason)
        Novelty: [1-5]/5 (reason)
    
    Each factor contributes 25% weight. Calculate panic probability considering:
        - Higher danger awareness, higher panic probability
        - Lower coping efficacy, higher panic probability
        - Higher uncertainty, higher panic probability
        - Higher novelty, higher panic probability
    Provide the final probability number in this exact format: [XX%] at the end
     """
                }
            ])
        
    # Phase 6: Tweet generation instructions
    elif phase == "tweet":
        # Build answer summary
        answer_summary = "\n".join(
            [f"{q}: score={ans['score']}, reason={ans['reason'][:2000]}..." 
             for q, ans in answers.items()]
        )
        # Ensure emotional_arousal_summary is a dictionary and correctly formatted
        if not isinstance(emotional_arousal_summary, dict):
            emotional_arousal_summary = {
                'danger_awareness': {'score': 3, 'reason': 'Default reason'},
                'coping_efficacy': {'score': 3, 'reason': 'Default reason'},
                'uncertainty': {'score': 3, 'reason': 'Default reason'},
                'novelty': {'score': 3, 'reason': 'Default reason'},
                'panic_probability': 50
            }
        
        # Format emotional_arousal_summary as readable string
        formatted_emotional_arousal = f"""
        Panic Factors Analysis:
        1. Awareness of Danger: Score {emotional_arousal_summary['danger_awareness']['score']}/5 ({emotional_arousal_summary['danger_awareness']['reason']})
        2. Coping Efficacy: Score {emotional_arousal_summary['coping_efficacy']['score']}/5 ({emotional_arousal_summary['coping_efficacy']['reason']})
        3. Uncertainty of Risk: Score {emotional_arousal_summary['uncertainty']['score']}/5 ({emotional_arousal_summary['uncertainty']['reason']})
        4. Novelty of Risk: Score {emotional_arousal_summary['novelty']['score']}/5 ({emotional_arousal_summary['novelty']['reason']})
        Panic Probability: {emotional_arousal_summary['panic_probability']}%

        """
        messages.extend([
            {
                "role": "user",
                "content": f"User risk perception assessment results: \n{answer_summary}"
            },
            {
                "role": "assistant", 
                "content": "Risk assessment integration completed"
            },
            {
                "role": "user",
                "content": f"Emotional arousal evaluation results: \n{formatted_emotional_arousal}"
            },
            {
                "role": "assistant", 
                "content": "Emotional arousal integration completed"
            },
            {
                "role": "user",
                "content": f"""After answering all questions, what text post would this user most likely publish during a hurricane? Generate tweet STRICTLY following these rules:
                [IMPORTANT INTEGRATION REQUIREMENTS]
                - Synthesize ALL previous analysis phases:
                  1). Psychology principles
                  2). Real-time hurricane data 
                  3). User's comprehensive trait data
                  4). Risk assessment scores and reasoning
                  5). Emotional arousal evaluation
                
                0. Generate EXACTLY 1 possible tweets
                1. Content MUST be enclosed in double quotes
                2. Hashtags should mirror user's style like #LaborRights 
                3. Use exactly this format: "[Tweet text with #hashtags]"
                4. End with ### End
                5. Consider the user's panic probability {emotional_arousal_summary['panic_probability']}% when generating tweets. A probability 49% <= Probability <=51% represents a neutral state between panic and calmness. The higher the probability (closer to 100%), the more likely panic emotions will be triggered; the lower the probability (closer to 0%), the less likely panic emotions will be triggered.
                6. If panic probability >51%, tweets should directly convey more panic、fear and anxiety, and tweets should include more emotional amplifiers, or more EMPHATIC capitalized words, or more repeated punctuation marks, or sensory details.
                7. If panic probability <49%, tweets should reflect more calmness and rationality, and tweets should show more composed language."""
                # If there's validation failure reason, pass it separately
                + (f"\nPrevious failure tweet and reason: {failure_reason}" if failure_reason else "")
            }
        ])
    # Phase 7: Text validity verification
    elif phase == "validation":
        validation_prompt = f"""You are a professional consistency evaluator. Please assess the user's newly generated text from the perspectives of psychology, linguistics, accuracy and emotion expression.
        The user's new comment is: "{tweet}"
        
        1. Psychological Validation: Check if the tweet aligns with the user's psychological profile.
        
        2. Linguistic Validation: Verify if the tweet's language style is consistent with the user's historical style.
            
        3. Factual Validation: Confirm if the tweet is relevant to Hurricane Sandy and factually accurate.
        
        4. Panic Probability Alignment
         [PANIC PROBABILITY]: Use the user's panic_probability value ({emotional_arousal_summary['panic_probability']}%) 
            - If >51% : Should show anxiety/fear/panic, and tweets should include more emotional amplifiers, or more EMPHATIC capitalized words, or more repeated punctuation marks, or sensory details.
            - If <49%: Should demonstrate calmness/rationality 
            - If 49% <= Probability <=51%: Neutral concern without panic
            
        [Response Format]
        Psychological: YES/NO (reason)
        Linguistic: YES/NO (reason) 
        Factual: YES/NO (reason)
        Panic: YES/NO (reason)"""
    
        messages.extend([
            {"role": "user", "content": validation_prompt},
            {"role": "assistant", "content": "Quadruple validation results"}
        ])
    return messages

'''

'''

def parse_tweet(response):
    """Extract tweet content from response text"""
    try:
        clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

        tweets = re.findall(
            r'^"([\s\S]+?)"\s*$',  
            clean_response,
            flags=re.MULTILINE
        )
        
        # Try loose matching if unsuccessful
        if not tweets:
            tweets = re.findall(r'"([^"]+)"', clean_response)
        
        cleaned_tweets = [re.sub(r'\s+', ' ', t).strip() for t in tweets][:1]
        
        # Verify quantity
        if len(cleaned_tweets) != 1:
            print(f"Warning: Parsed {len(cleaned_tweets)} tweets, expected 1")
            while len(cleaned_tweets) < 1:
                cleaned_tweets.append("Tweet parsing failed, please check original response")
        
        return cleaned_tweets[:1]  
    except Exception as e:
        print(f"Parsing error: {str(e)}")
        return [f"Parsing exception：{str(e)}"] * 1  

def parse_emotional_arousal(response_text):
    """Parse emotional arousal evaluation response"""
    emotional_arousal = {
        'danger_awareness': {'score': 3, 'reason': 'Default reason'},  
        'coping_efficacy': {'score': 3, 'reason': 'Default reason'},
        'uncertainty': {'score': 3, 'reason': 'Default reason'},
        'novelty': {'score': 3, 'reason': 'Default reason'},
        'panic_probability': 50
    }
    
    patterns = {
        'danger_awareness': r'Awareness:\s*(\d+)/5\s*\(([^)]+)',
        'coping_efficacy': r'Coping:\s*(\d+)/5\s*\(([^)]+)',
        'uncertainty': r'Uncertainty:\s*(\d+)/5\s*\(([^)]+)',
        'novelty': r'Novelty:\s*(\d+)/5\s*\(([^)]+)'
    }
    
    for factor, pattern in patterns.items():
        match = re.search(pattern, response_text)
        if match:
            emotional_arousal[factor]['score'] = int(match.group(1))
            emotional_arousal[factor]['reason'] = match.group(2).strip()
    # Parse panic probability
    panic_match = re.search(r'\[(\d+)%\]', response_text)
    if panic_match:
        emotional_arousal['panic_probability'] = int(panic_match.group(1))
    return emotional_arousal


def generate_valid_tweet(client, user_info, answers, emotional_arousal_data, max_retries=3):
    """Generate and validate a single tweet, max 3 retries per data point"""
    failure_reason = None  
    
    for _ in range(max_retries):
        # Generate single tweet
        tweet_messages = construct_conversation("tweet", user_info, answers, emotional_arousal_data, failure_reason)
        
        try:
            tweet_response = client.chat.completions.create(
                model="deepseek-chat", 
                messages=tweet_messages,
                temperature=0.7,  
                presence_penalty=0.4, 
                max_tokens=300,
                stop=["### End"]
            )
            raw_tweet = tweet_response.choices[0].message.content
            tweet = parse_tweet(raw_tweet)[0]  
            # print("Predicted text generation：\n", tweet)
            # Perform triple validation
            validation_messages = construct_conversation("validation", user_info, answers, emotional_arousal_data, failure_reason, tweet)
            validation_response = client.chat.completions.create(
                model="deepseek-chat", 
                messages=validation_messages,
                temperature=0.4,
                max_tokens=500
            )
            
            validation_result = validation_response.choices[0].message.content
            # print("Multi-expert evaluation：\n", validation_result)
            if "Psychological: YES" in validation_result and "Linguistic: YES" in validation_result and "Factual: YES" in validation_result and "Panic: YES" in validation_result:
                return tweet
            else:
                # Extract failure reasons
                failure_reasons = []
                failure_reasons.append(f"Previous tweet: {tweet}")
                if "Psychological: NO" in validation_result:
                    reason_match = re.search(r"Psychological: NO \(reason: (.*?)\)", validation_result)
                    if reason_match:
                        failure_reasons.append(f"Psychological: {reason_match.group(1)}")
                if "Linguistic: NO" in validation_result:
                    reason_match = re.search(r"Linguistic: NO \(reason: (.*?)\)", validation_result)
                    if reason_match:
                        failure_reasons.append(f"Linguistic: {reason_match.group(1)}")
                if "Factual: NO" in validation_result:
                    reason_match = re.search(r"Factual: NO \(reason: (.*?)\)", validation_result)
                    if reason_match:
                        failure_reasons.append(f"Factual: {reason_match.group(1)}")
                if "Panic: No" in validation_result:
                    reason_match = re.search(r"Panic: No \(reason: (.*?)\)", validation_result)
                    if reason_match:
                        failure_reasons.append(f"Panic: {reason_match.group(1)}")
                
                failure_reason = " | ".join(failure_reasons) if failure_reasons else "Validation failed for unknown reason"
                
                # Feed failure reason back to generation logic
                time.sleep(10)  
                continue
        
        except Exception as e:
            print(f"Tweet generation failed: {str(e)}")
            time.sleep(15)
    
    # Generate fallback content after max retries
    return f"[Emergency alert] Hurricane related update #WeatherAlert (Validation failed after {max_retries} attempts)"

def execute(user_info):
    client = OpenAI(
    api_key="your_api_key_here",
    base_url="https://api.deepseek.com"
                )

    try:
        # Phase 1: Get risk perception assessment
        assessment_messages = construct_conversation("assessment", user_info)
        assessment_response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=assessment_messages,
                    temperature=0.4, 
                    max_tokens=500,
                    stop=["### End"],  
                    
                )
        
        raw_assessment = assessment_response.choices[0].message.content
        # print("User：\n", user_info['user_id'])  
        # print("Raw risk perception response：\n", raw_assessment)  
        answers = parse_answers(raw_assessment)
        # print("Formatted risk perception response with questions：\n", answers)  
        # Phase 2: Emotional arousal evaluation
        emotional_arousal_messages = construct_conversation("emotional_arousal", user_info, answers)
        emotional_arousal_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=emotional_arousal_messages,
            temperature=0.4, 
            max_tokens=300,
            stop=["### End"],  
        )
        
        raw_emotional_arousal = emotional_arousal_response.choices[0].message.content
        # print("Emotional arousal evaluation：\n", raw_emotional_arousal)
        # Parse panic emotion arousal content and probability
        emotional_arousal_data = parse_emotional_arousal(raw_emotional_arousal)
        # print("Formatted emotional arousal evaluation：\n", emotional_arousal_data)
        
        # Phase 3: Generate tweets
        valid_tweets = []
        for _ in range(3):  # Generate 3 valid tweets
            tweet = generate_valid_tweet(client, user_info, answers, emotional_arousal_data)
            valid_tweets.append(tweet)
            time.sleep(15)  
        
        return valid_tweets
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

def process_user(row):
    try:
        user_info = build_user_profile(row)
        prediction = execute(user_info)
        return user_info['user_id'], prediction
    except Exception as e:
        print(f"Error processing user {row['user_id']}: {str(e)}")
        return None, None  # Return None to indicate processing failure

if __name__ == "__main__":
    data = load_data()
    results = []
    request_delay = 15

    with ThreadPoolExecutor(max_workers=4) as executor:  
        futures = [executor.submit(process_user, row) for _, row in data.iterrows()]
        for future in tqdm(futures, desc="Processing users"):
            try:
                user_id, prediction = future.result()
                if user_id is not None and prediction is not None:
                    for tweet in prediction:
                        results.append({"user_id": user_id, "predicative_text": tweet})
                # Save progress in real time
                pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
                time.sleep(request_delay)  
            except Exception as e:
                print(f"Error processing user: {str(e)}")
                time.sleep(request_delay * 2)
    
    print("Processing completed, results saved to:", OUTPUT_PATH)