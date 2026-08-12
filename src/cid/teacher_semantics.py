from __future__ import annotations

import re
from hashlib import sha256
from typing import Any

from cid.grounding import Anchor, AnchorKind, CognitiveLink, LinkRelation, ObjectRef

TEACHER_SEMANTIC_TEXT_MAX_CHARS = 192
TEACHER_SEMANTIC_TEXT_TARGET_CHARS = 144

_DATE_PATTERN = re.compile(
    r"(?:\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,)?\s+\d{3,4}\b"
    r"|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{3,4}\b)",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")

_NATIONALITY_WORDS = (
    "American",
    "Argentine",
    "Argentinean",
    "Argentinian",
    "Australian",
    "Austrian",
    "Belgian",
    "Brazilian",
    "British",
    "Bulgarian",
    "Canadian",
    "Chilean",
    "Chinese",
    "Colombian",
    "Croatian",
    "Czech",
    "Danish",
    "Dutch",
    "Egyptian",
    "English",
    "Finnish",
    "French",
    "German",
    "Ghanaian",
    "Greek",
    "Hungarian",
    "Icelandic",
    "Indian",
    "Indonesian",
    "Iranian",
    "Irish",
    "Israeli",
    "Italian",
    "Japanese",
    "Kuwaiti",
    "Laotian",
    "Lebanese",
    "Malaysian",
    "Mexican",
    "Nigerian",
    "Norwegian",
    "Pakistani",
    "Polish",
    "Portuguese",
    "Puerto Rican",
    "Romanian",
    "Russian",
    "Salvadoran",
    "Scottish",
    "Serbian",
    "Slovak",
    "Slovenian",
    "South African",
    "South Korean",
    "Spanish",
    "Swedish",
    "Swiss",
    "Syrian",
    "Thai",
    "Taiwanese",
    "Togolese",
    "Turkish",
    "Ugandan",
    "Ukrainian",
    "Venezuelan",
    "Vietnamese",
    "Welsh",
)
_COUNTRY_WORDS = (
    "Argentina",
    "Australia",
    "Austria",
    "Belgium",
    "Bosnia and Herzegovina",
    "Brazil",
    "Bulgaria",
    "Canada",
    "Chile",
    "China",
    "Colombia",
    "Croatia",
    "Czechia",
    "Denmark",
    "Egypt",
    "Finland",
    "France",
    "Germany",
    "Ghana",
    "Greece",
    "Guyana",
    "Hungary",
    "Hong Kong",
    "Iceland",
    "India",
    "Indonesia",
    "Iran",
    "Ireland",
    "Israel",
    "Italy",
    "Japan",
    "Kenya",
    "Kuwait",
    "Laos",
    "Lebanon",
    "Luxembourg",
    "Malaysia",
    "Maldives",
    "Mexico",
    "Netherlands",
    "New Zealand",
    "Nigeria",
    "Norway",
    "Pakistan",
    "Panama",
    "Poland",
    "Portugal",
    "Romania",
    "Russia",
    "Serbia",
    "Singapore",
    "Slovakia",
    "Slovenia",
    "South Africa",
    "South Korea",
    "Spain",
    "Sweden",
    "Switzerland",
    "Syria",
    "Thailand",
    "Turkey",
    "Uganda",
    "Ukraine",
    "United Kingdom",
    "United States",
    "Venezuela",
    "Vietnam",
)

_US_STATE_NAMES = (
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
)


def compact_text(text: str, *, limit: int = TEACHER_SEMANTIC_TEXT_MAX_CHARS) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    content_limit = max(1, limit - 1)
    cut = value[: content_limit + 1]
    boundary = max(cut.rfind("; "), cut.rfind(", "), cut.rfind(" "))
    cut = cut[:boundary] if boundary >= max(32, content_limit // 2) else cut[:content_limit]
    return cut.rstrip(" ,;:-") + "…"


def compact_task_intent(prompt: str) -> str:
    lowered = prompt.casefold()
    if "director" in lowered:
        if any(
            token in lowered for token in ("same country", "nationality", "from the same country")
        ):
            return "Need both director identities and nationalities."
        if any(
            token in lowered
            for token in ("born first", "born earlier", "born later", "older", "younger")
        ):
            return "Need both director identities and birth dates."
        if any(token in lowered for token in ("died first", "died earlier", "died later")):
            return "Need both director identities and death dates."
        if "place of birth" in lowered or ("where" in lowered and "born" in lowered):
            return "Need the director identity and birthplace."
        if "place of death" in lowered or ("where" in lowered and "die" in lowered):
            return "Need the director identity and death place."
        if "cause of death" in lowered or ("why" in lowered and "die" in lowered):
            return "Need the director identity and cause of death."
        if "birthday" in lowered or "date of birth" in lowered:
            return "Need the director identity and birth date."
        if "date of death" in lowered or "when did" in lowered and "die" in lowered:
            return "Need the director identity and death date."
        if "country" in lowered or "nationality" in lowered:
            return "Need the director identity and nationality."
    if "performer" in lowered or "song" in lowered:
        if "place of birth" in lowered or ("where" in lowered and "born" in lowered):
            return "Need the song performer identity and birthplace."
        if "nationality" in lowered or "country" in lowered:
            return "Need the performer identity and nationality."
        if "father" in lowered or "mother" in lowered:
            return "Need the performer identity and parent relation."
    if any(token in lowered for token in ("same country", "located in the same country")):
        return "Need the country for each compared location or entity."
    if any(
        token in lowered
        for token in ("released first", "released earlier", "came out earlier", "published first")
    ):
        return "Need release or publication dates for both compared items."
    if any(
        token in lowered
        for token in ("born first", "born earlier", "born later", "older", "younger")
    ):
        return "Need birth dates for both compared people."
    if any(token in lowered for token in ("died first", "died earlier", "died later")):
        return "Need death dates for both compared people."
    if any(
        token in lowered
        for token in (
            "father-in-law",
            "mother-in-law",
            "grandfather",
            "grandmother",
            "stepmother",
            "co-wife",
        )
    ):
        return "Need the relevant family relations to resolve the requested relative."
    if "graduate" in lowered or "studied" in lowered or "study" in lowered:
        return "Need the relevant person and education institution."
    if "established first" in lowered or "founded first" in lowered:
        return "Need establishment dates for both compared organizations."
    if "lived longer" in lowered:
        return "Need birth and death dates for both compared people."
    if "place of burial" in lowered or "buried" in lowered:
        return "Need the relevant person and burial place."
    if "work at" in lowered or "works at" in lowered:
        return "Need the relevant person and workplace."
    if "occupation" in lowered:
        return "Need occupations for the compared people."
    if "uncle" in lowered:
        return "Need parent and sibling relations to resolve the uncle."
    if "child" in lowered:
        return "Need the relevant parent-child relation."
    return "Need task-local evidence for the requested fact or relation."


def summarize_candidate_titles(titles: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    clean = tuple(dict.fromkeys(str(title) for title in titles if str(title).strip()))
    shown = clean[:5]
    summary = "Relevant records: " + "; ".join(shown) + ("; …" if len(clean) > len(shown) else ".")
    return compact_text(summary, limit=TEACHER_SEMANTIC_TEXT_TARGET_CHARS), clean


def summarize_search_results(value: Any) -> tuple[str, tuple[str, ...]]:
    titles: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                title = item.get("title") or item.get("resource_id")
                if title:
                    titles.append(str(title))
    return summarize_candidate_titles(tuple(titles))


def summarize_evidence(
    value: Any, evidence_id: str, task_prompt: str
) -> tuple[str, tuple[str, ...]]:
    title, text = _evidence_text(value, evidence_id)
    if not text:
        return compact_text(f"Evidence received for {title}."), (title,)

    prompt = task_prompt.casefold()
    relation = (
        _extract_subject_relation(text, prompt)
        if _title_is_named_subject(title, prompt) or _evidence_subject_is_media(text)
        else None
    )
    if relation is not None:
        name, fact = relation
        summary = compact_text(
            f"{title} — {name}: {fact}.", limit=TEACHER_SEMANTIC_TEXT_TARGET_CHARS
        )
        return summary, (title, fact)

    # For relational multi-hop questions, a document about the explicitly named subject
    # should preserve the outgoing relation before extracting the final-hop attribute.
    # Example: `When did Alfred Maynard's father die?` must first become
    # `Alfred Maynard — father: William Maynard`, not Alfred's own death date.
    if _asks_relation(prompt) and _title_is_named_subject(title, prompt):
        named_relation = _extract_named_subject_relation(text, prompt)
        if named_relation is not None:
            name, fact = named_relation
            summary = compact_text(
                f"{title} — {name}: {fact}.", limit=TEACHER_SEMANTIC_TEXT_TARGET_CHARS
            )
            return summary, (title, fact)

    fields: list[tuple[str, str]] = []

    language_comparison = (
        _extract_language_comparison(text) if _asks_language_comparison(prompt) else None
    )
    if language_comparison:
        fields.append(("language resembles", language_comparison))

    oldest_force_nationality = (
        _extract_oldest_force_nationality(text) if _asks_oldest_force_country(prompt) else None
    )
    if oldest_force_nationality:
        fields.append(("oldest force nationality", oldest_force_nationality))

    if _asks_birth_place(prompt):
        place = _extract_birth_place(text)
        if place:
            fields.append(("birth place", place))
    if _asks_birth_date(prompt):
        date = _extract_birth_date(text)
        if date:
            fields.append(("born", date))
    if _asks_death_place(prompt):
        place = _extract_death_place(text)
        if place:
            fields.append(("death place", place))
    if _asks_death_cause(prompt):
        cause = _extract_death_cause(text)
        if cause:
            fields.append(("death cause", cause))
    if _asks_death_date(prompt):
        date = _extract_death_date(text)
        if date:
            fields.append(("died", date))
    media_location_country = (
        _extract_media_location_country(text) if _asks_media_location_country(prompt) else None
    )
    if media_location_country:
        fields.append(("country", media_location_country))
    elif _asks_country(prompt) and not language_comparison and not oldest_force_nationality:
        first_sentence = _relation_sentences(text)[0]
        country_text = first_sentence if _looks_like_media_subject(first_sentence) else text
        if _asks_location_country(prompt):
            has_location_signal = _looks_like_media_subject(
                first_sentence
            ) or _has_subject_location_signal(first_sentence)
            # A complex question can contain `country where ... is located` even
            # while the current evidence is about an unrelated person/event.
            # Do not turn any incidental country mention in that person's prose
            # into a location answer.  Generic country scanning is appropriate
            # only when the evidence subject itself has a locative/media signal;
            # otherwise require an explicit origin relation.
            country = (
                _extract_country(country_text)
                if has_location_signal
                else _extract_origin_country(country_text)
            )
            if country:
                fields.append(("country", country))
            elif has_location_signal:
                nationality = _extract_nationality(country_text)
                if nationality:
                    fields.append(("nationality", nationality))
        else:
            nationality = _extract_nationality(country_text)
            if nationality:
                fields.append(("nationality", nationality))
            elif "nationality" not in prompt:
                country = (
                    _extract_country(country_text)
                    if _looks_like_media_subject(first_sentence)
                    else _extract_origin_country(country_text)
                )
                if country:
                    fields.append(("country", country))
    if _asks_release(prompt):
        released = _extract_release(text)
        if released:
            fields.append(("release", released))
    if _asks_education(prompt):
        school = _extract_education(text)
        if school:
            fields.append(("education", school))
    if _asks_establishment(prompt):
        established = _extract_establishment(text)
        if established:
            fields.append(("established", established))
    if _asks_burial(prompt):
        burial = _extract_burial(text)
        if burial:
            fields.append(("burial", burial))
    if _asks_workplace(prompt):
        workplace = _extract_workplace(text)
        if workplace:
            fields.append(("workplace", workplace))
    if _asks_occupation(prompt):
        occupation = _extract_occupation(text)
        if occupation:
            fields.append(("occupation", occupation))
    if _asks_relation(prompt) and not fields:
        relation = _extract_relation(text, prompt)
        if relation:
            fields.append(relation)
    if _asks_award(prompt):
        award = _extract_award(text)
        if award:
            fields.append(("award", award))

    # Preserve two task-relevant facts when the question is comparative or relational.
    unique_fields: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in fields:
        key = (item[0].casefold(), item[1].casefold())
        if key not in seen:
            seen.add(key)
            unique_fields.append(item)
    if unique_fields:
        summary = (
            f"{title} — " + "; ".join(f"{name}: {fact}" for name, fact in unique_fields[:3]) + "."
        )
        return compact_text(summary, limit=TEACHER_SEMANTIC_TEXT_TARGET_CHARS), (
            title,
            *(fact for _, fact in unique_fields[:2]),
        )

    missing_field = _requested_missing_field(prompt)
    if missing_field is not None:
        summary = f"{title} — {missing_field}: not stated in visible evidence."
        return compact_text(summary, limit=TEACHER_SEMANTIC_TEXT_TARGET_CHARS), (title,)

    fallback = _fallback_fact(text, title)
    return compact_text(f"{title} — {fallback}", limit=TEACHER_SEMANTIC_TEXT_TARGET_CHARS), (title,)


def entity_anchor(value: str, *, confidence: float = 1.0) -> Anchor:
    canonical = " ".join(value.split()).strip()
    digest = sha256(canonical.casefold().encode("utf-8")).hexdigest()[:16]
    return Anchor(
        anchor_id=f"entity:{digest}",
        kind=AnchorKind.ENTITY,
        value=canonical,
        object_id=f"entity:{canonical.casefold()}",
        confidence=confidence,
    )


def text_anchor(value: str, *, confidence: float = 1.0) -> Anchor:
    canonical = " ".join(value.split()).strip()
    digest = sha256(canonical.casefold().encode("utf-8")).hexdigest()[:16]
    return Anchor(
        anchor_id=f"text:{digest}",
        kind=AnchorKind.TEXT,
        value=canonical,
        confidence=confidence,
    )


def anchors_for_values(values: tuple[str, ...], *, max_anchors: int = 3) -> tuple[Anchor, ...]:
    anchors: list[Anchor] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        canonical = " ".join(str(value).split()).strip(" .")
        if not canonical or canonical.casefold() in {"yes", "no"}:
            continue
        key = canonical.casefold()
        if key in seen:
            continue
        seen.add(key)
        anchor = (
            entity_anchor(canonical)
            if index == 0 or _looks_entity_like(canonical)
            else text_anchor(canonical)
        )
        anchors.append(anchor)
        if len(anchors) >= max_anchors:
            break
    return tuple(anchors)


def source_link(relation: LinkRelation, source: str, *, confidence: float = 1.0) -> CognitiveLink:
    return CognitiveLink(relation=relation, target=ObjectRef.source(source), confidence=confidence)


def cell_link(relation: LinkRelation, cell_id: str, *, confidence: float = 1.0) -> CognitiveLink:
    return CognitiveLink(relation=relation, target=ObjectRef.cell(cell_id), confidence=confidence)


def answer_anchor(answer: str) -> tuple[Anchor, ...]:
    value = " ".join(answer.split()).strip(" .")
    if (
        not value
        or value.casefold() in {"yes", "no"}
        or value.casefold().startswith("the visible evidence")
    ):
        return ()
    return (text_anchor(value),)


def _evidence_text(value: Any, evidence_id: str) -> tuple[str, str]:
    if isinstance(value, dict):
        title = str(value.get("title") or value.get("resource_id") or evidence_id)
        sentences = value.get("sentences")
        if isinstance(sentences, list):
            return title, "\n".join(str(item).strip() for item in sentences if str(item).strip())
        return title, str(value.get("text") or value.get("value") or "")
    if isinstance(value, str):
        return evidence_id, value
    return evidence_id, str(value or "")


def _title_is_named_subject(title: str, prompt: str) -> bool:
    def normalize(value: str) -> str:
        return " ".join(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))

    normalized_title = normalize(title)
    normalized_prompt = normalize(prompt)
    return bool(normalized_title) and normalized_title in normalized_prompt


def _looks_like_media_subject(sentence: str) -> bool:
    lowered = sentence.casefold()
    if re.search(r"\b(?:film|movie|documentary|song|album|series) director\b", lowered):
        return False
    return bool(re.search(r"\b(?:film|movie|documentary|song|album|series)\b", lowered))


def _has_subject_location_signal(sentence: str) -> bool:
    """Whether the lead clause explicitly locates the evidence subject."""

    return bool(
        re.search(
            r"\b(?:located|situated|based|headquartered|born|raised|founded|formed)\b"
            r"[^.;]{0,100}\b(?:in|at|on)\b",
            sentence,
            re.IGNORECASE,
        )
        or re.search(
            r"\bis\b[^.;]{0,120}\b(?:in|on)\b",
            sentence,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:operates?|serves?|owns?|runs?|maintains?)\b[^.;]{0,100}"
            r"\b(?:U\.?S\.?|United States)\b(?!\$)",
            sentence,
            re.IGNORECASE,
        )
    )


def _evidence_subject_is_media(text: str) -> bool:
    first = _relation_sentences(text)[0] if text.strip() else ""
    lowered = first.casefold()
    if re.search(r"\b(?:film|movie|documentary|song|album|series) director\b", lowered):
        return False
    return bool(
        re.search(
            r"\b(?:film|movie|documentary|song|album|series)\b.{0,180}"
            r"\b(?:directed|written and directed|recorded|performed|composed|produced) by\b",
            lowered,
        )
    )


def _asks_birth_place(prompt: str) -> bool:
    return (
        "place of birth" in prompt
        or "born in the same place" in prompt
        or "same birth place" in prompt
        or "where was" in prompt
        and "born" in prompt
        or "where did" in prompt
        and "born" in prompt
    )


def _asks_birth_date(prompt: str) -> bool:
    return any(
        token in prompt
        for token in (
            "birthday",
            "date of birth",
            "born first",
            "born earlier",
            "born later",
            "older",
            "younger",
            "lived longer",
        )
    ) or bool(re.search(r"\bwhen\s+(?:was|were)\b[^?]{0,160}\bborn\b", prompt))


def _asks_death_place(prompt: str) -> bool:
    return (
        "place of death" in prompt
        or "where did" in prompt
        and " die" in prompt
        or "where was" in prompt
        and "death" in prompt
    )


def _asks_death_cause(prompt: str) -> bool:
    return "cause of death" in prompt or "why did" in prompt and " die" in prompt


def _asks_death_date(prompt: str) -> bool:
    return (
        "date of death" in prompt
        or "died first" in prompt
        or "died earlier" in prompt
        or "died later" in prompt
        or "lived longer" in prompt
        or ("when did" in prompt and " die" in prompt)
    )


def _asks_country(prompt: str) -> bool:
    return any(
        token in prompt
        for token in (
            "nationality",
            "which country",
            "what country",
            "same country",
            "from the same country",
            "located in the same country",
            "located in which country",
            "country is from",
            "country the",
            "country where",
        )
    )


def _asks_location_country(prompt: str) -> bool:
    return "located" in prompt or "location" in prompt


def _asks_media_location_country(prompt: str) -> bool:
    return "country" in prompt and any(
        token in prompt
        for token in (
            "filmed",
            "shot",
            "takes place",
            "take place",
            "set in",
        )
    )


def _asks_language_comparison(prompt: str) -> bool:
    return "language" in prompt and any(
        token in prompt for token in ("resembles", "resemble", "sounded like", "sounds like")
    )


def _asks_oldest_force_country(prompt: str) -> bool:
    return "oldest" in prompt and any(token in prompt for token in ("navy", "marine force"))


def _asks_release(prompt: str) -> bool:
    return any(
        token in prompt
        for token in (
            "released first",
            "released earlier",
            "came out earlier",
            "came out first",
            "published first",
            "was released",
            "release date",
        )
    )


def _asks_establishment(prompt: str) -> bool:
    return any(
        token in prompt
        for token in (
            "established first",
            "founded first",
            "established earlier",
            "founded earlier",
        )
    )


def _asks_burial(prompt: str) -> bool:
    return "place of burial" in prompt or "where was" in prompt and "buried" in prompt


def _asks_workplace(prompt: str) -> bool:
    return "work at" in prompt or "works at" in prompt or "worked at" in prompt


def _asks_occupation(prompt: str) -> bool:
    return "occupation" in prompt


def _asks_education(prompt: str) -> bool:
    return any(
        token in prompt
        for token in ("graduate", "graduated", "study", "studied", "education", "school did")
    )


def _asks_relation(prompt: str) -> bool:
    return any(
        token in prompt
        for token in (
            "father",
            "mother",
            "husband",
            "wife",
            "spouse",
            "grandfather",
            "grandmother",
            "father-in-law",
            "mother-in-law",
            "co-wife",
            "stepmother",
            "uncle",
            "child",
        )
    )


def _asks_award(prompt: str) -> bool:
    return any(token in prompt for token in ("award", "prize", "honor", "honour"))


def _extract_subject_relation(text: str, prompt: str) -> tuple[str, str] | None:
    patterns: list[tuple[str, str]] = []
    if "director" in prompt:
        patterns.extend(
            [
                ("director", r"\bdirected,?\s+produced by and starring\s+(.+)$"),
                ("director", r"\bdirected by and starring\s+(.+)$"),
                ("director", r"\bwritten and directed by\s+(.+)$"),
                (
                    "director",
                    r"\bdirected by\s+(?:filmmaker\s+)?([^,;]+?)(?=,|;|$)",
                ),
                ("director", r"\bdirected\s+(?!by\b)(.+)$"),
                ("director", r"\bdirected.{0,45}?\bby\s+(.+)$"),
                ("director", r"\bfrom directors?\s+(.+)$"),
                ("director", r"^Director\s+(.+?)(?=\s+(?:was|is)\b|,|;|$)"),
                ("director", r"\b(?:film|documentary) by (?:filmmaker and activist )?(.+)$"),
            ]
        )
    if "composer" in prompt:
        patterns.extend(
            [
                ("composer", r"\bcomposed by\s+([^,;]+?)(?=,|;|$)"),
                (
                    "composer",
                    r"\b(?:music|score|musical score)\s+(?:(?:was|is)\s+)?"
                    r"(?:composed )?by\s+(.+)$",
                ),
                ("composer", r"\bcomposed by\s+(.+)$"),
                ("composer", r"\b(?:show tune|composition)\s+by\s+(.+)$"),
                ("composer", r"\bwritten and produced by\s+(.+)$"),
                ("composer", r"\bmusic score provided by\s+(.+)$"),
                ("composer", r"\bmusic from\s+(.+)$"),
                ("composer", r"\bby\s+(.+?)['’]s (?:musical )?score\b"),
                ("composer", r"\badaptation of\s+(.+?)['’]s (?:classic )?opera\b"),
                ("composer", r"^(.+?)\s+provided (?:the )?(?:film['’]s )?soundtrack\b"),
            ]
        )
        if "song" in prompt:
            patterns.append(
                (
                    "composer",
                    r"\b(?:song\s+)?(?:was\s+)?written by\s+"
                    r"(?:lead singer\s+)?(.+)$",
                )
            )
    if "performer" in prompt or (
        "song" in prompt
        and "composer" not in prompt
        and "producer" not in prompt
        and not _asks_release(prompt)
    ):
        patterns.extend(
            [
                ("performer", r"\balbum by\s+([^,;]+?)(?=,|;|$)"),
                # Prefer the subject credit in sentences such as
                # `"Miss O'Dell" is a song by English musician George Harrison,
                # released as the B-side of ... "Give Me Love".`  The broader
                # `song by ...` fallback below can otherwise drift to a later
                # title in the same sentence when relation cleanup searches for
                # the final named entity.
                ("performer", r"\bsong by\s+([^,;]+?)(?=,|;|$)"),
                ("performer", r"\bsong (?:first )?recorded by\s+([^,.;]+?)(?=[,.;]|$)"),
                ("performer", r"\brecorded by\s+([^,.;]+?)(?=[,.;]|$)"),
                ("performer", r"\brecorded(?: on .{1,40})? by\s+(.+)$"),
                ("performer", r"\b(?:song|single)\s+(?:was\s+)?performed by\s+(.+)$"),
                ("performer", r"\band performed by\s+(.+)$"),
                ("performer", r"\bsingle by\s+(.+)$"),
                ("performer", r"\bhit for\s+(.+)$"),
                ("performer", r"\bsong by\s+(.+)$"),
            ]
        )
    if "producer" in prompt:
        patterns.append(("producer", r"\bproduced by\s+(.+)$"))
    if "creator" in prompt:
        patterns.extend(
            [
                ("creator", r"\bcreated and hosted by\s+(.+)$"),
                ("creator", r"\bcreated by\s+(.+)$"),
            ]
        )
    if "founder" in prompt:
        patterns.extend(
            [
                ("founder", r"\bfounder(?: and CEO)?[, ]+(.+)$"),
                ("founder", r"\bfounded by\s+(.+)$"),
            ]
        )

    for sentence in _relation_sentences(text):
        for name, pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if not match:
                continue
            raw_value = _clean_relation_fact(match.group(1))
            candidate = _clean_relation_entity(raw_value)
            if name in {"composer", "director", "performer"} and " and " in raw_value:
                enumerated = [
                    _clean_relation_entity(part)
                    for part in re.split(r",\s*|\s+and\s+", raw_value)
                    if part.strip()
                ]
                if 2 <= len(enumerated) <= 4 and all(
                    _plausible_relation_entity(part) for part in enumerated
                ):
                    return name, ", ".join(enumerated[:-1]) + " and " + enumerated[-1]
                left, right = raw_value.split(" and ", 1)
                role_words = re.compile(
                    r"\b(?:filmmaker|journalist|actor|actress|writer|producer|director|"
                    r"singer|musician|screenwriter|cinematographer)\b",
                    re.IGNORECASE,
                )
                left_entity = _clean_relation_entity(left)
                right_entity = _clean_relation_entity(right)
                if (
                    not role_words.search(left)
                    and not role_words.search(right)
                    and _plausible_relation_entity(left_entity)
                    and _plausible_relation_entity(right_entity)
                ):
                    candidate = f"{left_entity} and {right_entity}"
            if _plausible_relation_entity(candidate):
                return name, candidate

    if "composer" in prompt and "song" in prompt:
        for sentence in _relation_sentences(text):
            match = re.search(r"\bsong by\s+(.+)$", sentence, re.IGNORECASE)
            if not match:
                continue
            candidate = _clean_relation_entity(_clean_relation_fact(match.group(1)))
            if _plausible_relation_entity(candidate):
                return "composer", candidate
    return None


def _requested_missing_field(prompt: str) -> str | None:
    if _asks_birth_place(prompt):
        return "birth place"
    if _asks_birth_date(prompt):
        return "birth date"
    if _asks_death_place(prompt):
        return "death place"
    if _asks_death_cause(prompt):
        return "death cause"
    if _asks_death_date(prompt):
        return "death date"
    if _asks_country(prompt):
        return "country/nationality"
    if _asks_release(prompt):
        return "release/publication date"
    if _asks_education(prompt):
        return "education"
    if _asks_establishment(prompt):
        return "establishment date"
    if _asks_burial(prompt):
        return "burial place"
    if _asks_workplace(prompt):
        return "workplace"
    if _asks_occupation(prompt):
        return "occupation"
    if _asks_award(prompt):
        return "award"
    if _asks_relation(prompt):
        return "requested relation"
    return None


def _extract_birth_date(text: str) -> str | None:
    match = re.search(r"\bborn(?:\s+on)?\s+([^.;]{3,40})", text, re.IGNORECASE)
    if match:
        date = _first_date(match.group(1))
        if date:
            return date
        year = _YEAR_PATTERN.search(match.group(1))
        if year:
            return year.group(0)
    malformed_pair = re.search(
        r"(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{4})(?=\d{1,2}\s+(?:January|February|March|"
        r"April|May|June|July|August|September|October|November|December)\s+\d{4})",
        text[:320],
        re.IGNORECASE,
    )
    if malformed_pair:
        return malformed_pair.group(1)
    year_lifespan = re.search(
        r"\(\s*(\d{3,4})\s*[–—-]\s*(?:" + _DATE_PATTERN.pattern + r"|\d{3,4})",
        text[:320],
        re.IGNORECASE,
    )
    if year_lifespan:
        return year_lifespan.group(1)
    dates = _DATE_PATTERN.findall(text[:260])
    if dates:
        return dates[0].strip(" ,")
    return None


def _extract_death_date(text: str) -> str | None:
    match = re.search(r"\bdied(?:\s+on)?\s+([^.;]{3,60})", text, re.IGNORECASE)
    if match:
        date = _first_date(match.group(1))
        if date:
            return date
    missing_birth = re.search(
        r"\(\s*[–—-]\s*(" + _DATE_PATTERN.pattern + r")",
        text[:320],
        re.IGNORECASE,
    )
    if missing_birth:
        return missing_birth.group(1).strip(" ,")
    malformed_pair = re.search(
        r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{4}(\d{1,2}\s+(?:January|February|March|"
        r"April|May|June|July|August|September|October|November|December)\s+\d{4})",
        text[:320],
        re.IGNORECASE,
    )
    if malformed_pair:
        return malformed_pair.group(1)
    year_then_date = re.search(
        r"\(\s*\d{3,4}\s*[–—-]\s*(" + _DATE_PATTERN.pattern + r")",
        text[:320],
        re.IGNORECASE,
    )
    if year_then_date:
        return year_then_date.group(1).strip(" ,")
    dates = _DATE_PATTERN.findall(text[:320])
    if len(dates) >= 2:
        return dates[1].strip(" ,")
    return None


def _extract_birth_place(text: str) -> str | None:
    for sentence in _relation_sentences(text):
        match = re.search(r"\bborn\b.{0,80}?\bin\s+(.+)$", sentence, re.IGNORECASE)
        if match:
            location = _trim_location(match.group(1))
            if _plausible_location(location):
                return location
        match = re.search(r"\bborn\s+at\s+(.+)$", sentence, re.IGNORECASE)
        if match:
            location = _trim_location(match.group(1))
            if _plausible_location(location):
                return location
        if not re.search(r"\b(?:died|death|assassinated|killed)\b", sentence, re.IGNORECASE):
            match = re.search(
                r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+\d{3,4}\s+in\s+([^–—;.)]+)",
                sentence,
                re.IGNORECASE,
            )
            if match:
                location = _trim_location(match.group(1))
                if _plausible_location(location):
                    return location
    parenthetical_place_date = re.search(
        r"\(\s*([^()]{3,100}?)\s+(?=(?:January|February|March|April|May|June|July|"
        r"August|September|October|November|December)\s+\d{1,2},?\s+\d{4})",
        text[:420],
        re.IGNORECASE,
    )
    if parenthetical_place_date:
        location = _trim_location(parenthetical_place_date.group(1))
        if _plausible_location(location):
            return location
    lifespan_place = re.search(
        r"\(\s*([^,()]+,\s*[^,()]+),\s*(?:" + _DATE_PATTERN.pattern + r")\s*[–—-]",
        text[:360],
        re.IGNORECASE,
    )
    if lifespan_place:
        location = _trim_location(lifespan_place.group(1))
        if _plausible_location(location):
            return location
    return None


def _extract_death_place(text: str) -> str | None:
    for sentence in _relation_sentences(text):
        match = re.search(r"\bdied\b.{0,100}?\bin\s+(.+)$", sentence, re.IGNORECASE)
        if match:
            location = _trim_location(match.group(1))
            if _plausible_location(location):
                return location
        match = re.search(r"\bdied\s+at\s+(.+)$", sentence, re.IGNORECASE)
        if match:
            location = _trim_location(match.group(1))
            if _plausible_location(location):
                return location
        match = re.search(
            r"\bassassinated(?:\s+on)?\s+[^.;]{0,60}?\bin\s+(.+)$",
            sentence,
            re.IGNORECASE,
        )
        if match:
            location = _trim_location(match.group(1))
            if _plausible_location(location):
                return location
    assassination = re.search(
        r"\bassassinated(?:\s+on)?\s+[^.;]{0,60}?\bin\s+(.+)$",
        text,
        re.IGNORECASE,
    )
    if assassination:
        location = _trim_location(assassination.group(1))
        if _plausible_location(location):
            return location
    date_in_place = re.search(
        r"[–—-]\s*(?:" + _DATE_PATTERN.pattern + r")\s+in\s+([^)]+)\)",
        text[:360],
        re.IGNORECASE,
    )
    if date_in_place:
        location = _trim_location(date_in_place.group(1))
        if _plausible_location(location):
            return location
    location_then_date = re.search(
        r"[–—-]\s*([^,()]+,\s*[^,()]+),\s*(?:" + _DATE_PATTERN.pattern + r")\s*\)",
        text[:360],
        re.IGNORECASE,
    )
    if location_then_date:
        location = _trim_location(location_then_date.group(1))
        if _plausible_location(location):
            return location
    lifespan_place = re.search(
        r"\([^)]*[–—-]\s*(?:" + _DATE_PATTERN.pattern + r"|\d{3,4})\s*,\s*([^)]+)\)",
        text[:360],
        re.IGNORECASE,
    )
    if lifespan_place:
        location = _trim_location(lifespan_place.group(1))
        if _plausible_location(location):
            return location
    return None


def _extract_death_cause(text: str) -> str | None:
    relative_subject = re.compile(
        r"\b(?:his|her|their)\s+(?:husband|wife|father|mother|son|daughter|brother|sister)\b",
        re.IGNORECASE,
    )
    for sentence in _relation_sentences(text):
        if relative_subject.search(sentence) and re.search(r"\bdied\b", sentence, re.IGNORECASE):
            continue
        match = re.search(
            r"\bdied[^.;]{0,100}?\b(?:from|of)\s+([^.;,]{3,80})",
            sentence,
            re.IGNORECASE,
        )
        if match:
            value = re.split(
                r"\s+(?:during|after|while|following|at)\b",
                match.group(1),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            return _clean_fact(value)
        match = re.search(
            r"\b(?:became|was)\s+ill\s+with\s+([^.;,]{3,100}?)(?=,?\s+and\s+died\b)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return _clean_fact(match.group(1))
    return None


def _extract_media_location_country(text: str) -> str | None:
    for sentence in _relation_sentences(text)[:4]:
        for pattern in (
            r"\b(?:shot|filmed)\s+in\s+(.+)$",
            r"\bfilming (?:took place|occurred)\s+in\s+(.+)$",
            r"\bset\s+in\s+(.+)$",
            r"\btakes? place\s+in\s+(.+)$",
        ):
            match = re.search(pattern, sentence, re.IGNORECASE)
            if not match:
                continue
            country = _extract_country(match.group(1)[:180])
            if country:
                return country
    return None


def _extract_language_comparison(text: str) -> str | None:
    for sentence in _relation_sentences(text)[:4]:
        match = re.search(
            r"\b(?:sounded|sounds|sound)\s+like\s+(?:the\s+)?language\s+of\s+(?:the\s+)?"
            r"([A-ZÀ-ÖØ-Þ][\w'’.-]+)",
            sentence,
            re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r"\blanguage\b[^.;]{0,80}\b(?:resembles?|resembled)\s+"
                r"(?:the\s+)?(?:language\s+of\s+)?(?:the\s+)?([A-ZÀ-ÖØ-Þ][\w'’.-]+)",
                sentence,
                re.IGNORECASE,
            )
        if not match:
            continue
        value = _clean_fact(match.group(1))
        if value.casefold() == "persians":
            return "Persian"
        return value
    return None


def _extract_oldest_force_nationality(text: str) -> str | None:
    nationality_alt = "|".join(
        re.escape(word) for word in sorted(_NATIONALITY_WORDS, key=len, reverse=True)
    )
    for sentence in _relation_sentences(text)[:4]:
        match = re.search(
            rf"\b(?:The\s+)?({nationality_alt})\b[^.;]{{0,140}}"
            r"\b(?:oldest|oldest, current)\b",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).title()
    return None


def _extract_nationality(text: str) -> str | None:
    # Use the first sentence that actually states a nationality. This keeps a
    # subject's identity separate from later mentions of foreign collaborators,
    # locations, or industries.
    for sentence in _relation_sentences(text)[:4]:
        sentence = re.split(
            r"\b(?:consisting of|whose members include|featuring members)\b",
            sentence,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        sentence = re.sub(r"\bBritish Columbia\b", "", sentence, flags=re.IGNORECASE)
        sentence = re.sub(
            r"\bEnglish\s+(?:festival\s+title|version|title|translation)\b[^)]{0,120}\)?",
            "",
            sentence,
            flags=re.IGNORECASE,
        )
        matches: list[tuple[int, int, str, bool]] = []
        for word in _NATIONALITY_WORDS:
            for match in re.finditer(rf"\b{re.escape(word)}\b", sentence, re.IGNORECASE):
                prefix = sentence[: match.start()]
                suffix = sentence[match.end() :]
                identity_prefix = bool(
                    re.search(
                        r"\b(?:is|was|are|were)\s+(?:an?\s+)?(?:\d{4}\s+)?"
                        r"(?:(?:former|retired|professional|pioneering|prominent|leading|"
                        r"well-known|renowned|famous|successful|noted|veteran)\s+){0,3}$",
                        prefix[-40:],
                        re.IGNORECASE,
                    )
                )
                work_modifier = bool(
                    re.match(
                        r"\s+(?:television\s+serials?|films?|movies?|cinema|industry)\b",
                        suffix,
                        re.IGNORECASE,
                    )
                )
                language_theatre = bool(
                    re.search(r"\bprofessional\s+$", prefix[-32:], re.IGNORECASE)
                    and re.match(r"\s+(?:and\s+\w+\s+)?theatre\b", suffix, re.IGNORECASE)
                )
                ancestry_context = bool(
                    re.search(
                        r"\b(?:father|mother|parent|parents|grandfather|grandmother|family)(?:'s|s')?[^.;]{0,32}$",
                        prefix[-72:],
                        re.IGNORECASE,
                    )
                    or re.match(
                        r"\s+(?:roots|ancestry|descent|heritage|background|origin)\b",
                        suffix,
                        re.IGNORECASE,
                    )
                    or re.search(
                        r"\b(?:son|daughter|child)\s+of\b[^.;]{0,80}$",
                        prefix[-120:],
                        re.IGNORECASE,
                    )
                )
                non_subject_modifier = bool(
                    re.match(
                        r"\s+(?:colon(?:y|ies)|troops?|forces?|arm(?:y|ies)|rule|occupation|"
                        r"territor(?:y|ies))\b",
                        suffix,
                        re.IGNORECASE,
                    )
                )
                compound_continuation = prefix.endswith("-") and any(
                    re.search(rf"\b{re.escape(other)}-$", prefix, re.IGNORECASE)
                    for other in _NATIONALITY_WORDS
                    if other != word
                )
                if (
                    work_modifier and not identity_prefix and not compound_continuation
                ) or language_theatre:
                    continue
                if ancestry_context and not identity_prefix:
                    continue
                if non_subject_modifier and not identity_prefix:
                    continue
                matches.append((match.start(), match.end(), word, identity_prefix))
        if not matches:
            continue
        matches.sort(key=lambda item: item[0])
        identity_indexes = [index for index, item in enumerate(matches) if item[3]]
        if identity_indexes:
            keep = set(identity_indexes)
            # Preserve true compound identities such as `Canadian-American` when
            # one side is attached to the subject-defining copula. Do not also
            # absorb unrelated nationality adjectives later in the sentence.
            changed = True
            while changed:
                changed = False
                for index in tuple(keep):
                    if (
                        index > 0
                        and sentence[matches[index - 1][1] : matches[index][0]].strip() == "-"
                        and index - 1 not in keep
                    ):
                        keep.add(index - 1)
                        changed = True
                    if (
                        index + 1 < len(matches)
                        and sentence[matches[index][1] : matches[index + 1][0]].strip() == "-"
                        and index + 1 not in keep
                    ):
                        keep.add(index + 1)
                        changed = True
                    if (
                        index + 1 < len(matches)
                        and re.fullmatch(
                            r"-born\s+",
                            sentence[matches[index][1] : matches[index + 1][0]],
                            re.IGNORECASE,
                        )
                        and index + 1 not in keep
                    ):
                        keep.add(index + 1)
                        changed = True
            matches = [item for index, item in enumerate(matches) if index in keep]
        elif len(matches) > 1:
            # Multiple nationality adjectives can describe unrelated people or
            # objects in the same evidence sentence. Do not synthesize a compound
            # nationality unless the words are literally hyphen-linked.
            compound = all(
                sentence[left[1] : right[0]].strip() == "-"
                for left, right in zip(matches, matches[1:], strict=False)
            )
            if not compound:
                return None
        ordered: list[str] = []
        for _, _, word, _ in matches:
            if word not in ordered:
                ordered.append(word)
        return "-".join(ordered[:2]) if len(ordered) > 1 else ordered[0]
    return None


def _extract_country(text: str) -> str | None:
    for sentence in _relation_sentences(text)[:4]:
        if re.search(r"\b(?:U\.?S\.?|United States)\b(?!\$)", sentence, re.IGNORECASE):
            return "United States"
        for state in sorted(_US_STATE_NAMES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(state)}\b", sentence, re.IGNORECASE):
                return "United States"
        matches: list[tuple[int, int, str]] = []
        for country in _COUNTRY_WORDS:
            match = re.search(rf"\b{re.escape(country)}\b", sentence, re.IGNORECASE)
            if match:
                matches.append((match.start(), -len(country), country))
        if matches:
            matches.sort()
            return matches[0][2]
    return None


def _extract_origin_country(text: str) -> str | None:
    """Return only a country explicitly tied to the person's origin.

    Generic mentions such as working in a country's film industry are not evidence
    that the person is from that country.
    """

    for sentence in _relation_sentences(text)[:4]:
        for country in sorted(_COUNTRY_WORDS, key=len, reverse=True):
            if re.search(
                rf"\b(?:is|was|are|were)\s+(?:an?\s+)?(?:\d{{4}}\s+)?{re.escape(country)}\b",
                sentence,
                re.IGNORECASE,
            ):
                return country
        born = re.search(r"\bborn\b(.+)$", sentence, re.IGNORECASE)
        if born:
            country = _extract_country(born.group(1))
            if country:
                return country
        origin = re.search(
            r"\b(?:from|native of|a native of)\s+(.+)$",
            sentence,
            re.IGNORECASE,
        )
        if origin:
            country = _extract_country(origin.group(1))
            if country:
                return country
    return None


def _extract_release(text: str) -> str | None:
    match = re.search(
        r"\b(?:released|premiered|published|established|founded)\b[^.;]{0,90}", text, re.IGNORECASE
    )
    if match:
        date = _first_date(match.group(0))
        if date:
            return date
        year = _YEAR_PATTERN.search(match.group(0))
        if year:
            return year.group(0)
    year = _YEAR_PATTERN.search(text[:180])
    return year.group(0) if year else None


def _extract_education(text: str) -> str | None:
    educated = re.search(
        r"\beducated at\s+([^.;,]{3,100}), followed by\s+([^.;]{3,100}?)(?:, he |;|\.)",
        text,
        re.IGNORECASE,
    )
    if not educated:
        educated = re.search(
            r"\beducated at\s+([^.;]{3,100}?)(?:,|;|\.)",
            text,
            re.IGNORECASE,
        )
    if educated:
        schools = [_clean_fact(value) for value in educated.groups() if value]
        return "; ".join(schools)
    for pattern in (
        r"\bgraduated from\s+([^.;]{3,100})",
        r"\bgraduated at\s+([^.;]{3,100})",
        r"\bstudied at\s+([^.;]{3,100})",
        r"\bdegree(?: of [^.;]+)? from\s+([^.;]{3,100})",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _clean_fact(match.group(1))
    return None


def _extract_establishment(text: str) -> str | None:
    match = re.search(
        r"\b(?:founded|established|created|opened|launched)\b[^.;]{0,100}", text, re.IGNORECASE
    )
    if not match:
        return None
    date = _first_date(match.group(0))
    if date:
        return date
    year = _YEAR_PATTERN.search(match.group(0))
    return year.group(0) if year else None


def _extract_burial(text: str) -> str | None:
    for pattern in (
        r"\bburied (?:at|in|on)\s+([^.;]{3,100})",
        r"\binterred (?:at|in)\s+([^.;]{3,100})",
        r"\bgrave is located at\s+([^.;]{3,100})",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _clean_fact(match.group(1))
    return None


def _extract_workplace(text: str) -> str | None:
    # Prefer a named publication/institution over a broad movement or field.
    match = re.search(r"\bfilm magazine [\"“]?([^\"”.;]{3,80})", text, re.IGNORECASE)
    if match:
        return _clean_fact(match.group(1))
    match = re.search(
        r"\bhired by\s+[^.;]{0,80}?\bfor [\"“]?([^\"”.;]{3,80})",
        text,
        re.IGNORECASE,
    )
    if match:
        return _clean_fact(match.group(1))
    patterns = (
        r"\bworked at\s+([^.;]{3,100})",
        r"\bworks at\s+([^.;]{3,100})",
        r"\b(?:editor|editor-in-chief|professor) (?:of|at)\s+([^.;]{3,100})",
        r"\bassociated with\s+([^.;]{3,100})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _clean_fact(match.group(1))
    return None


def _extract_occupation(text: str) -> str | None:
    match = re.search(r"\b(?:is|was) an?\s+([^.;]{2,100})", text, re.IGNORECASE)
    if not match:
        return None
    value = re.split(
        r"\s+(?:with|who|known|from|based|specializing|specialising|and (?:a|an|the) )",
        match.group(1),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = _clean_fact(value)
    for nationality in sorted(_NATIONALITY_WORDS, key=len, reverse=True):
        prefix = nationality + " "
        if value.casefold().startswith(prefix.casefold()):
            value = value[len(prefix) :].strip()
            break
    return value or None


def _extract_named_subject_relation(text: str, prompt: str) -> tuple[str, str] | None:
    """Extract the outgoing relation from a subject explicitly named by the task."""
    if "father-in-law" in prompt or "mother-in-law" in prompt:
        spouse = _extract_spouse(text)
        return None if spouse is None else ("spouse", spouse)
    if "uncle" in prompt:
        parent = _extract_parent(text, "father") or _extract_parent(text, "mother")
        return None if parent is None else ("father", parent)
    if "paternal grandfather" in prompt or "paternal grandmother" in prompt:
        parent = _extract_parent(text, "father")
        return None if parent is None else ("father", parent)
    if "maternal grandfather" in prompt or "maternal grandmother" in prompt:
        parent = _extract_parent(text, "mother")
        return None if parent is None else ("mother", parent)
    if "grandfather" in prompt or "grandmother" in prompt:
        father = _extract_parent(text, "father")
        if father is not None:
            return "father", father
        mother = _extract_parent(text, "mother")
        return None if mother is None else ("mother", mother)
    if "father" in prompt:
        father = _extract_parent(text, "father")
        return None if father is None else ("father", father)
    if "mother" in prompt:
        mother = _extract_parent(text, "mother")
        return None if mother is None else ("mother", mother)
    if any(token in prompt for token in ("wife", "husband", "spouse", "co-wife")):
        spouse = _extract_spouse(text)
        return None if spouse is None else ("spouse", spouse)
    if "child" in prompt:
        child = _extract_child(text)
        return None if child is None else ("child", child)
    return None


def _extract_relation(text: str, prompt: str) -> tuple[str, str] | None:
    """Extract the next relation for a non-subject hop."""
    if "father-in-law" in prompt:
        father = _extract_parent(text, "father")
        return None if father is None else ("father", father)
    if "mother-in-law" in prompt:
        mother = _extract_parent(text, "mother")
        return None if mother is None else ("mother", mother)
    if "uncle" in prompt:
        brother = _extract_brother(text)
        return None if brother is None else ("brother", brother)
    if "child" in prompt:
        child = _extract_child(text)
        if child is not None:
            return "child", child
    if "paternal grandfather" in prompt or "maternal grandfather" in prompt:
        father = _extract_parent(text, "father")
        return None if father is None else ("father", father)
    if "paternal grandmother" in prompt or "maternal grandmother" in prompt:
        mother = _extract_parent(text, "mother")
        return None if mother is None else ("mother", mother)
    if "grandfather" in prompt:
        father = _extract_parent(text, "father")
        if father is not None:
            return "father", father
    if "grandmother" in prompt:
        mother = _extract_parent(text, "mother")
        if mother is not None:
            return "mother", mother
    if "father" in prompt:
        father = _extract_parent(text, "father")
        if father is not None:
            return "father", father
    if "mother" in prompt:
        mother = _extract_parent(text, "mother")
        if mother is not None:
            return "mother", mother
    if any(token in prompt for token in ("wife", "husband", "spouse", "co-wife")):
        spouse = _extract_spouse(text)
        if spouse is not None:
            return "spouse", spouse
    return None


def _relation_sentences(text: str) -> tuple[str, ...]:
    lines = tuple(" ".join(line.split()) for line in text.splitlines() if line.strip())
    return lines or (" ".join(text.split()),)


def _plausible_relation_entity(value: str | None) -> bool:
    if not value:
        return False
    low = value.casefold().strip(" .")
    if low in {
        "he",
        "she",
        "his",
        "her",
        "in",
        "on",
        "at",
        "the",
        "a",
        "an",
    }:
        return False
    return bool(re.search(r"[A-ZÀ-ÖØ-Þ]", value))


def _clean_parent_candidate(value: str) -> str | None:
    value = " ".join(value.split()).strip(" ,.;:-")
    nationality_alt = "|".join(re.escape(word) for word in _NATIONALITY_WORDS)
    value = re.sub(
        rf"^(?:(?:{nationality_alt})(?:\s+and\s+(?:{nationality_alt}))?)\s+"
        r"(?:King|Queen|Prince|Princess|Duke|Duchess|Archduke|Archduchess)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^(?:his|her|their)(?: (?:first|second|third|fourth))? "
        r"(?:wife(?: and niece)?|spouse)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^former\s+(?:President|First Lady)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^.*\b(?:football|soccer|hurling|baseball|basketball|cricket)?\s*player\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    parts = [part.strip() for part in value.split(",")]
    first = parts[0] if parts else value
    if len(parts) > 1:
        suffix = re.match(r"^(Jr|Sr)\.", parts[1], re.IGNORECASE)
        if suffix:
            first = f"{first} {suffix.group(1)}."
    if len(parts) > 1 and re.match(
        r"^(?:(?:\d+(?:st|nd|rd|th))\s+)?(?:Count|Duke|Marquess|Earl|Prince|"
        r"Princess|King|Queen|Emperor|Baron|Viscount|Lord|Lady)\b",
        parts[1],
        re.IGNORECASE,
    ):
        first_candidate = _clean_relation_entity(first)
        candidate = (
            compact_text(f"{first_candidate}, {parts[1]}", limit=96)
            if _plausible_relation_entity(first_candidate)
            else None
        )
    else:
        first_candidate = _clean_relation_entity(first)
        candidate = (
            first_candidate
            if _plausible_relation_entity(first_candidate)
            else _clean_relation_entity(value)
        )
    return candidate if _plausible_relation_entity(candidate) else None


def _extract_parent_pair(clause: str) -> tuple[str | None, str | None]:
    clause = " ".join(clause.split()).strip(" ,.;:-")
    clause = re.split(
        r",\s+(?:but|who|which|he|she|they|it|one of\b|"
        r"and (?:was|became|succeeded|followed|later|then))\b",
        clause,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    clause = re.split(
        r"\s+and\s+(?:a|an|the)?\s*(?:member of|niece|nephew|grandson|granddaughter|"
        r"younger brother|younger sister|older brother|older sister)\b",
        clause,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    clause = re.split(
        r",\s+(?:the\s+)?(?:son|daughter|child) of\b",
        clause,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    spouse_marker = re.search(
        r"\s+(?:and|by)\s+(?:his|her|their)(?: (?:first|second|third|fourth))? "
        r"(?:[A-Za-zÀ-ÖØ-öø-ÿ-]+-born\s+)?(?:wife(?: and niece)?|spouse),?\s+(.+)$",
        clause,
        re.IGNORECASE,
    )
    if spouse_marker:
        father_raw = clause[: spouse_marker.start()].strip(" ,")
        mother_raw = spouse_marker.group(1).strip(" ,")
        return _clean_parent_candidate(father_raw), _clean_parent_candidate(mother_raw)

    # Parent order is not always father-then-mother.  When exactly one side
    # carries an explicitly female kinship/title/occupation marker, use that
    # marker rather than positional order.  This covers formulations such as
    # `daughter of Swedish actress Ingrid Bergman and ... Roberto Rossellini`.
    for separator in reversed(list(re.finditer(r"\s+and\s+", clause, re.IGNORECASE))):
        left_raw = clause[: separator.start()]
        right_raw = clause[separator.end() :]
        female_marker = re.compile(
            r"\b(?:actress|mother|wife|queen|princess|duchess|archduchess)\b",
            re.IGNORECASE,
        )
        left_female = bool(female_marker.search(left_raw))
        right_female = bool(female_marker.search(right_raw))
        if left_female == right_female:
            continue
        mother_raw = left_raw if left_female else right_raw
        father_raw = right_raw if left_female else left_raw
        father = _clean_parent_candidate(father_raw)
        mother = _clean_parent_candidate(mother_raw)
        if father and mother:
            return father, mother

    comma_pair = re.match(r"^(.+?),\s+and\s+(.+)$", clause, re.IGNORECASE)
    if comma_pair:
        father = _clean_parent_candidate(comma_pair.group(1))
        mother = _clean_parent_candidate(comma_pair.group(2))
        if father and mother:
            return father, mother

    for separator in reversed(list(re.finditer(r"\s+and\s+", clause, re.IGNORECASE))):
        left = clause[: separator.start()]
        right = clause[separator.end() :]
        father = _clean_parent_candidate(left)
        mother = _clean_parent_candidate(right)
        if father and mother:
            return father, mother

    return _clean_parent_candidate(clause), None


def _extract_parent(text: str, relation: str) -> str | None:
    if relation not in {"father", "mother"}:
        raise ValueError(f"unsupported parent relation: {relation}")

    if relation == "father":
        for sentence in _relation_sentences(text):
            match = re.search(
                r"\b(?:predecessor\s+and\s+)?father,\s+(.+)$",
                sentence,
                re.IGNORECASE,
            )
            if match:
                candidate = _clean_parent_candidate(match.group(1))
                if candidate:
                    return candidate

    born_to = re.compile(r"\bborn\b[^.;]{0,100}?\bto\s+(.+)$", re.IGNORECASE)
    for sentence in _relation_sentences(text):
        match = born_to.search(sentence)
        if not match:
            continue
        parent_clause = match.group(1).strip()
        if re.match(r"^(?:the\s+)?(?:musical\s+)?family of\b", parent_clause, re.IGNORECASE):
            continue
        father, mother = _extract_parent_pair(parent_clause)
        if relation == "father" and father:
            return father
        if relation == "mother" and mother:
            return mother

    # Kinship clauses are more reliable than later narrative mentions such as
    # `his father fled...` or `succeeded his father in...`.
    kinship = re.compile(
        r"\b(?:sons?(?:\s+and\s+(?:successor|heir))?|daughters?|"
        r"child(?:ren)?(?:\s+and\s+heir)?)\s+of\s+(.+)$",
        re.IGNORECASE,
    )
    for sentence in _relation_sentences(text):
        match = kinship.search(sentence)
        if not match:
            continue
        parent_clause = match.group(1)
        if relation == "father":
            role_parent = re.search(
                r"\b(?:football|soccer|hurling|baseball|basketball|cricket)?\s*player\s+(.+)$",
                parent_clause,
                re.IGNORECASE,
            )
            if role_parent:
                candidate = _clean_parent_candidate(role_parent.group(1))
                if candidate:
                    return candidate
        father, mother = _extract_parent_pair(parent_clause)
        if relation == "father" and father:
            raw = match.group(1).casefold()
            if (
                any(
                    token in raw
                    for token in ("mistress to", "wife of", "mother of", "queen ", "princess ")
                )
                and mother is None
            ):
                continue
            return father
        if relation == "mother" and mother:
            return mother

    explicit = re.compile(
        rf"\b(?:his|her|their)\s+{relation}(?:\s+was|\s+is|,)?\s+(.+)$",
        re.IGNORECASE,
    )
    rejected_prefixes = (
        "fled ",
        "died ",
        "in ",
        "into ",
        "at ",
        "from ",
        "had ",
        "has ",
        "succeeded ",
        "followed ",
        "left ",
    )
    for sentence in _relation_sentences(text):
        match = explicit.search(sentence)
        if not match:
            continue
        phrase = match.group(1).strip()
        if phrase.casefold().startswith(rejected_prefixes):
            continue
        phrase = re.split(
            r"\s+and\s+(?:his|her|their)\s+",
            phrase,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        candidate = _clean_parent_candidate(phrase)
        if candidate:
            return candidate
    return None


def _extract_spouse(text: str) -> str | None:
    patterns = (
        r"\b(?:his|her) husband was\s+(.+)$",
        r"\b(?:his|her) wife was\s+(.+)$",
        r"\bmarried to the [^,;]+,\s+(.+)$",
        r"\bwith (?:his|her) wife\s+(.+)$",
        r"\b(?:his wife|her husband)\s+(.+?)(?=\s+(?:was|is|prepared|worked|served|"
        r"became|died|born|helped|had)\b|,|;|\.|$)",
        r"\bmarried (?:his|her) (?:third |second |first )?wife\s+(.+)$",
        r"\bcivil partnership with\s+(.+)$",
        r"\blongtime companion of\s+(.+)$",
        r"\bmarriage to\s+(.+)$",
        r"\bmarried to\s+(.+?)(?:\s+of the \d|$)",
        r"\bmarried\s+(?!(?:from|since|in)\b)(.+?)(?=\.\s+(?:Their|They|He|She|His|Her|The)\b|;|$)",
        r"\b(?:was|is).{0,140}\b(?:wife|wives|husband|spouse) of\s+(.+)$",
        r"\bas (?:the )?(?:first |second |third )?(?:wife|husband|spouse) of\s+(.+)$",
        r"\bas well as the (?:wife|husband) of\s+(.+)$",
        r"\band the (?:wife|husband) of\s+(.+)$",
        r"^(?:the )?(?:first|second|third)(?: and former)? wife of\s+(.+)$",
    )
    for sentence in _relation_sentences(text):
        for pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if not match:
                continue
            candidate = _clean_relation_entity(match.group(1))
            if _plausible_relation_entity(candidate):
                return candidate
    return None


def _extract_child(text: str) -> str | None:
    patterns = (
        r"\bfather of (?:the )?(?:actor |actress |director |writer |singer )?([^.;,]{3,90})",
        r"\bmother of (?:the )?(?:actor |actress |director |writer |singer )?([^.;,]{3,90})",
        r"\b(?:eldest|oldest|youngest|second) son(?: is)?\s+(.+?)(?=\s+is\b|,|;|$)",
        r"\b(?:the couple were|they were) the parents of\s+(.+?)(?=,|;|\.|$)",
        r"\b(?:their|his|her) (?:son|daughter)\s+(.+?)(?=\s+is\b|,|;|\.|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _clean_relation_entity(match.group(1))
    return None


def _extract_brother(text: str) -> str | None:
    patterns = (
        r"\b(?:elder |younger )?brother\s+([^.;,]{3,90})",
        r"\b(?:elder |younger )?brother (?:is|was)\s+([^.;,]{3,90})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _clean_relation_entity(match.group(1))
    return None


def _extract_award(text: str) -> str | None:
    number = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
    named_family = (
        r"(?:Grammy|American Music|MTV Video Music|Academy|Golden Globe|Filmfare|"
        r"BAFTA|Emmy|Tony)"
    )
    for sentence in _relation_sentences(text):
        match = re.search(
            rf"\b({number}\s+{named_family}\s+Awards?)\b",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return _clean_fact(match.group(1))
    for sentence in _relation_sentences(text):
        match = re.search(
            r"\b(?:was\s+)?awarded\s+(?:the\s+)?"
            r"([A-ZÀ-ÖØ-Þ][^.;,]{2,80}?)(?=\s+(?:in\s+\d{4}|by\s+)|[.;,]|$)",
            sentence,
        )
        if match:
            return _clean_fact(match.group(1))
    for sentence in _relation_sentences(text):
        match = re.search(
            r"\b(?:recipient of|won|received|awarded|garnered)\s+(?:the\s+)?"
            r"([^.;,]{2,100}?\b(?:Awards|Award|Prizes|Prize|Medals|Medal)"
            r"(?:\s+for\s+[^.;,]{2,70})?)",
            sentence,
            re.IGNORECASE,
        )
        if match:
            return _clean_fact(match.group(1))
    return None


def _fallback_fact(text: str, title: str) -> str:
    first = re.split(r"(?<=[.!?])\s+", " ".join(text.split()), maxsplit=1)[0]
    escaped = re.escape(title)
    first = re.sub(
        rf"^{escaped}\s*(?:\([^)]*\))?\s*(?:is|was|were)?\s*", "", first, flags=re.IGNORECASE
    )
    first = re.sub(r"\s+", " ", first).strip(" ,.;:-")
    if len(first) > 132:
        # Prefer the lead clause over a copied full sentence.
        clause = re.split(r"[,;]", first, maxsplit=1)[0].strip()
        if 12 <= len(clause) <= 132:
            first = clause
    return compact_text(first or "task-relevant evidence received", limit=144).rstrip(".") + "."


def _first_date(text: str) -> str | None:
    match = _DATE_PATTERN.search(text)
    return match.group(0).strip(" ,") if match else None


def _trim_location(value: str) -> str:
    value = value.split("--", 1)[0]
    value = re.sub(r"^(?:in|at)\s+", "", value, flags=re.IGNORECASE)
    if ")" in value:
        value = value[: value.index(")") + 1] if "(" in value else value.split(")", 1)[0]
    value = re.split(
        r",\s+(?=(?:the\s+)?[A-ZÀ-ÖØ-Þ][^,]{0,60}\b"
        r"(?:was|is|made|developed|initially|worked|pursued|became|later|studied|rose|showed)\b)",
        value,
        maxsplit=1,
    )[0]
    value = re.split(
        r"\s+on\s+(?:the\s+)?\d{1,2}\s+"
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = re.split(
        r"\b(?:and|where|who|which|before|after|at age|from|to)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = re.split(r"\s+as\s+[A-ZÀ-ÖØ-Þ]", value, maxsplit=1)[0]
    value = re.split(r"\s+of a heart attack\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
    value = re.split(
        r"\s+in\s+\d{4}\b|\s+at\s+the\s+age\b|\s+at\s+age\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return _clean_fact(value)


def _plausible_location(value: str | None) -> bool:
    if not value:
        return False
    # A lifespan can be captured by the loose parenthetical-place fallback,
    # e.g. `(June 20, 1811 – April 10, 1882)`.  A calendar date is never a
    # plausible geographic answer here.
    if _DATE_PATTERN.search(value):
        return False
    if re.fullmatch(
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{4}",
        value.strip(" .,"),
        re.IGNORECASE,
    ):
        return False
    if value.casefold().strip(" .") in {
        "infancy",
        "childhood",
        "exile",
        "prison",
        "captivity",
        "home",
    }:
        return False
    return bool(re.search(r"[A-ZÀ-ÖØ-Þ]", value))


def _clean_relation_entity(value: str) -> str:
    value = _clean_relation_fact(value)
    state_alt = "|".join(
        re.escape(state) for state in sorted(_US_STATE_NAMES, key=len, reverse=True)
    )
    value = re.sub(
        rf"^(?:former\s+)?Governor of (?:{state_alt})\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    numeric_name = re.match(
        r"^(\d+\s+[A-ZÀ-ÖØ-Þ][\w'’.-]*(?:\s+[A-ZÀ-ÖØ-Þ][\w'’.-]*){0,3})\b",
        value,
    )
    if numeric_name:
        return compact_text(numeric_name.group(1), limit=96).rstrip("…")
    nationality_alt = "|".join(re.escape(word) for word in _NATIONALITY_WORDS)
    value = re.sub(
        rf"^(?:{nationality_alt})\s+(?=[A-ZÀ-ÖØ-Þ])",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+until\b.*$", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r",\s+(Jr|Sr)\.", r" \1.", value)
    if "(" in value:
        base = value.split("(", 1)[0].strip()
        if base:
            value = base
    token = r"(?:[A-ZÀ-ÖØ-Þ][\w'’.-]*|[A-Z]\.|[IVXLCDM]+)"
    connector = r"(?:de|da|del|di|du|van|von|of|in|the)"
    pattern = re.compile(rf"{token}(?:\s+(?:{token}|{connector}))*")
    candidates = [match.group(0).strip() for match in pattern.finditer(value)]
    ampersand_act = re.fullmatch(
        r"[A-ZÀ-ÖØ-Þ][\w'’.-]*(?:\s+[A-ZÀ-ÖØ-Þ][\w'’.-]*)*\s*&\s*"
        r"[A-ZÀ-ÖØ-Þ][\w'’.-]*(?:\s+[A-ZÀ-ÖØ-Þ][\w'’.-]*)*",
        value,
    )
    if ampersand_act:
        return compact_text(value, limit=96).rstrip("…")
    if "," in value:
        first_clause = value.split(",", 1)[0].strip()
        first_candidates = [match.group(0).strip() for match in pattern.finditer(first_clause)]
        descriptive_first = bool(
            re.search(
                r"\b(?:actor|actress|director|writer|producer|singer|musician|politician|"
                r"lawyer|poet|novelist|filmmaker)\b",
                first_clause,
                re.IGNORECASE,
            )
        )
        if first_candidates and first_clause[:1].isupper() and not descriptive_first:
            value = first_candidates[-1]
        elif candidates:
            value = candidates[-1]
    elif candidates:
        value = candidates[-1]
    value = re.sub(
        r"^(?:U\.S\.\s+)?(?:Congressman|Senator|President|Vice President|First Lady|"
        r"Governor(?: of [A-ZÀ-ÖØ-Þ][\w'’.-]*)?|actor|actress|writer|poet|novelist|"
        r"filmmaker|director|leader)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^Dr\.\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(?:Roman\s+)?Emperor\s+", "", value)
    return compact_text(value.strip(" ,.;:-"), limit=96).rstrip("…")


def _clean_relation_fact(value: str) -> str:
    value = " ".join(value.split()).strip(" ,.;:-")
    value = re.split(
        r",\s*(?:(?:co-)?written|produced|distributed|starring|starred|released|financed|"
        r"lyrics)\s+by\b|,?\s+with whom\b|,?\s+and managed by\b|"
        r",?\s+and the songs are sung by\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = re.split(
        r"\s+(?:and (?:(?:co-)?written|directed|produced|distributed|starring|starred|"
        r"released|financed|featuring|recorded|performed|stars|features)(?: by)?|"
        r"starring\b|starred\b|stars\b|features\b|"
        r"serving as\b|during\b|with whom\b|as an? adaptation\b|"
        r"and the cinematography by\b|and managed by\b|"
        r"and served\b|and shot\b|and filmed\b|and (?:his|her|their) orchestra\b|"
        r"and (?:the )?(?:son|daughter|child|mother|father)\b|from a screenplay|"
        r"focusing on|about\b|"
        r"based (?:on|upon)|from \d{4}|in \d{4}|for (?=[A-Z0-9])|with vocal\b|"
        r"with (?=[A-ZÀ-ÖØ-Þ])|featuring\b|and stars\b|"
        r"on (?:his|her|their|the)\b|from (?:his|her|their|the)\b|in (?:his|her|their)\b|"
        r",\s*(?:(?:co-)?written|produced|distributed|starring|starred|released|financed) by\b|"
        r",?\s+and (?:the )?(?:mother|father) of\b|"
        r"\(on the\b|before\b|is |was |has |have |who |whom |which |that )",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return compact_text(value, limit=96).rstrip("…")


def _clean_fact(value: str) -> str:
    value = " ".join(value.split()).strip(" ,.;:-")
    return compact_text(value, limit=96).rstrip("…")


def _looks_entity_like(value: str) -> bool:
    return bool(re.search(r"[A-Za-z]", value)) and not bool(_DATE_PATTERN.fullmatch(value))
