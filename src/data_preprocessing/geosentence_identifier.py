import re
from rapidfuzz import fuzz
from src.utils.keyword_loader import load_keywords_from_txt_directory

GEOGRAPHIC_KEYWORDS = load_keywords_from_txt_directory("data/keywords/")

def normalize_text(text):
    """Normalize text by removing extra spaces and converting to lowercase."""
    return re.sub(r"\s+", " ", text.strip().lower())

def match_entity_in_sentence(sentence, entity_dict, entity_threshold=90):
    """Match entities from the entity dictionary in a sentence."""
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

def match_keywords_in_sentence(sentence, keyword_set=GEOGRAPHIC_KEYWORDS, keyword_threshold=90):
    """Match geographic keywords in a sentence."""
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
    keyword_matches = match_keywords_in_sentence(sentence, GEOGRAPHIC_KEYWORDS, keyword_threshold)
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
        results.append(classification)

    return results
