import re
from rapidfuzz import fuzz
from src.utils.keyword_loader import load_keywords_from_txt_directory

KEYWORDS_LIST = load_keywords_from_txt_directory("data/keywords/")

def normalize_text(text):
    """Normalize text by removing extra spaces and converting to lowercase."""
    return re.sub(r"\s+", " ", text.strip().lower())

# get fuzzy match index of phrase in text
def get_fuzzy_match_index(text, phrase, score):
    best_score = 0
    best_start = -1
    best_end = -1
    best_sub_text = ""
    return_score = 0
    for i in range(len(text) - len(phrase) + 1):
        sub_text = text[i:i+len(phrase)]
        temp_score = fuzz.partial_ratio(sub_text.lower(), phrase.lower())
        if temp_score > best_score:
            best_score = temp_score
            best_start = i
            best_sub_text = sub_text
            best_end = i + len(phrase)
        if temp_score == score:
            # best_score = score
            return_score = temp_score
            best_start = i
            best_sub_text = sub_text
            best_end = i + len(phrase)
            return best_start, best_end, return_score, best_sub_text

    # for match in re.finditer(rf'\b{re.escape(phrase)}\b', text, re.IGNORECASE):
    #     temp_score = fuzz.partial_ratio(match.group().lower(), phrase.lower())
    #     if temp_score > best_score:
    #         best_score = temp_score
    #         best_sub_text = match.group()
    #         best_start, best_end = match.start(), match.end()
    #     if temp_score == score:
    #         return_score = temp_score
    #         best_sub_text = match.group()
    #         best_start, best_end = match.start(), match.end()
    #         return best_start, best_end, return_score, best_sub_text
    return best_start, best_end, best_score, best_sub_text

def match_entity_in_sentence(sentence, entity_dict, entity_threshold=90):
    """
        Match entities in selected columns of the geojson attributes (here I call this entity dictionary) 
        in a sentence.

    """
    sent_norm = normalize_text(sentence)
    matches = []
    for label, phrases in entity_dict.items():
        if isinstance(phrases, str):  # Support for str or list[str]
            phrases = [phrases]
        for phrase in phrases:
            if not isinstance(phrase, str) or not phrase.strip():
                continue
            phrase_norm = normalize_text(phrase)
            score = fuzz.partial_ratio(sent_norm, phrase_norm)
            if score >= entity_threshold:
                matches.append((label, phrase, score))
                break  # Stop once a match is found
    return matches

def match_entity_in_sentence_for_subd_label(sentence, entity_dict, entity_threshold=90):
    """
        Fuzzy match subdivision entities from the entity dictionary in a sentence.
        This function is used for generate subdivision labels from geojson file for NER training data preparation.
    """
    sent_norm = normalize_text(sentence)
    matches = []
    for label, phrases in entity_dict.items():
        if isinstance(phrases, str):  # Support for str or list[str]
            phrases = [phrases]
        for phrase in phrases:
            if not isinstance(phrase, str) or not phrase.strip():
                continue
            phrase_norm = normalize_text(phrase)
            score = fuzz.partial_ratio(sent_norm, phrase_norm)
            
            if score >= entity_threshold:
                # print("sentence:", sentence)
                # print("sent_norm:", sent_norm)
                # print("phrase:", phrase)
             
                start_index = sent_norm.find(phrase)
                end_index = start_index + len(phrase)
                if score < 100:
                    best_start, best_end, matched_score, best_sub_text = get_fuzzy_match_index(sent_norm, phrase, score)
                    print("best_start:", best_start)
                    print("best_end:", best_end)
                    print("matched_score:", matched_score)
                    print("best_sub_text:", sent_norm[best_start:best_end])
                    matches.append((best_start, best_end, label, phrase, score))
                    break  # Stop once a match is found  
                if start_index == -1:
                    print(f"Warning: '{phrase}' not found in generated text.")
                    # return None
                    end_index = start_index
                matches.append((start_index, end_index, label, phrase, score))
                break  # Stop once a match is found   
    return matches

def match_keywords_in_sentence(sentence, keyword_set=KEYWORDS_LIST, keyword_threshold=90):
    """Match keywords from the keywords directory in a sentence."""
    sentence_words = set(re.findall(r"\b\w+\b", sentence.lower()))
    matches = []
    for kw in keyword_set:
        for word in sentence_words:
            score = fuzz.ratio(word, kw.lower())
            if score >= keyword_threshold:
                matches.append((kw, word, score))
                break
    return matches

def classify_sentence(sentence, entity_dict, entity_threshold=90, keyword_threshold=90):
    """Classify a sentence as GEO or NON-GEO based on entity and keyword matches."""
    entity_matches = match_entity_in_sentence(sentence, entity_dict, entity_threshold)
    keyword_matches = match_keywords_in_sentence(sentence, KEYWORDS_LIST, keyword_threshold)
    is_geo = bool(entity_matches or keyword_matches)
    return {
        "sentence": sentence,
        "geo_class": "GEO" if is_geo else "NON-GEO",
        "entity_matches": entity_matches,
        "keyword_matches": keyword_matches,
    }

def filter_relevant_sentences(text, entity_dict={}, entity_threshold=90, keyword_threshold=90, nlp=None):
    """Filter sentences in the text that match geographic entities or keywords."""
    doc = nlp(text)  # Assuming `nlp` is passed externally to avoid reloading the model inside the function

    clean_entity_dict = {
        k: normalize_text(v)
        for k, v in entity_dict.items()
        if isinstance(v, str) and v.strip() != ""
    }

    results = []
    for sent in doc.sents:
        classification = classify_sentence(sent.text.strip(), clean_entity_dict, entity_threshold, keyword_threshold)
        if classification['geo_class'] in ['GEO']: # in ['GEO', 'NON-GEO'] to print both
            results.append(classification)

    return results
