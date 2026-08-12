from cid.teacher_semantics import (
    TEACHER_SEMANTIC_TEXT_MAX_CHARS,
    compact_task_intent,
    compact_text,
    summarize_candidate_titles,
    summarize_evidence,
)


def test_compact_task_intent_is_state_not_prompt_copy() -> None:
    prompt = (
        "Do both films Wild Man Blues and The Kid from Kokomo have directors from the same country?"
    )
    intent = compact_task_intent(prompt)

    assert intent == "Need both director identities and nationalities."
    assert "Wild Man Blues" not in intent


def test_evidence_summary_uses_task_fact_slots() -> None:
    film, film_anchors = summarize_evidence(
        {
            "title": "Wild Man Blues",
            "sentences": [
                "Wild Man Blues is a 1997 documentary film directed by Barbara Kopple, "
                "about the musical avocation of Woody Allen."
            ],
        },
        "support-film",
        "Are the directors of both films from the same country?",
    )
    person, person_anchors = summarize_evidence(
        {
            "title": "Barbara Kopple",
            "sentences": [
                "Barbara Kopple (born July 30, 1946) is an American film director "
                "known primarily for documentary work."
            ],
        },
        "support-person",
        "Are the directors of both films from the same country?",
    )

    assert film == "Wild Man Blues — director: Barbara Kopple."
    assert film_anchors == ("Wild Man Blues", "Barbara Kopple")
    assert person == "Barbara Kopple — nationality: American."
    assert person_anchors == ("Barbara Kopple", "American")


def test_missing_requested_fact_is_explicit_not_source_prose() -> None:
    summary, anchors = summarize_evidence(
        {
            "title": "Elvis Presley",
            "sentences": [
                "Elvis Presley was an American singer and actor.",
                "He was born in Tupelo, Mississippi.",
            ],
        },
        "support-person",
        "Where was the place of burial of the performer of the song?",
    )

    assert summary == "Elvis Presley — burial place: not stated in visible evidence."
    assert anchors == ("Elvis Presley",)


def test_search_summary_keeps_only_selected_records() -> None:
    summary, anchors = summarize_candidate_titles(
        ("Wild Man Blues", "The Kid from Kokomo", "Barbara Kopple", "Lewis Seiler")
    )

    assert summary == (
        "Relevant records: Wild Man Blues; The Kid from Kokomo; Barbara Kopple; Lewis Seiler."
    )
    assert anchors == (
        "Wild Man Blues",
        "The Kid from Kokomo",
        "Barbara Kopple",
        "Lewis Seiler",
    )


def test_compact_text_includes_ellipsis_inside_hard_limit() -> None:
    value = compact_text("x" * 1000)

    assert len(value) == TEACHER_SEMANTIC_TEXT_MAX_CHARS
    assert value.endswith("…")


def test_occupation_and_multihop_relation_slots_do_not_fall_back_to_prose() -> None:
    occupation, _ = summarize_evidence(
        {
            "title": "Leonardo Salgado",
            "sentences": [
                "Leonardo Salgado is an Argentinean palaeontologist with a special "
                "interest in dinosaurs."
            ],
        },
        "support-person",
        "Do Leonardo Salgado and Alana de la Garza have the same occupation?",
    )
    parent, _ = summarize_evidence(
        {
            "title": "Boris Kovalchuk",
            "sentences": ["He is a son of Yury Kovalchuk."],
        },
        "support-person",
        "Who is the uncle of Boris Kovalchuk?",
    )
    sibling, _ = summarize_evidence(
        {
            "title": "Yury Kovalchuk",
            "sentences": [
                "His elder brother Mikhail Kovalchuk is the scientific secretary of a council."
            ],
        },
        "support-person",
        "Who is the uncle of Boris Kovalchuk?",
    )
    child, _ = summarize_evidence(
        {
            "title": "Master Venu",
            "sentences": [
                "Master Venu was a music composer and the father of the actor Bhanu Chander."
            ],
        },
        "support-person",
        "Who is the child of the composer of the film?",
    )

    assert occupation == "Leonardo Salgado — occupation: palaeontologist."
    assert parent == "Boris Kovalchuk — father: Yury Kovalchuk."
    assert sibling == "Yury Kovalchuk — brother: Mikhail Kovalchuk."
    assert child == "Master Venu — child: Bhanu Chander."


def test_country_slot_uses_first_entity_defining_country_not_later_distractor() -> None:
    summary, _ = summarize_evidence(
        {
            "title": "Western Springs, Illinois",
            "sentences": [
                "Western Springs is a village located in Cook County, Illinois, United States "
                "and is a suburb of Chicago.",
                "It is twinned with Rugeley, United Kingdom.",
            ],
        },
        "support-location",
        "Are Western Springs, Illinois and Hazerswoude-Rijndijk both located in the same country?",
    )

    assert summary == "Western Springs, Illinois — country: United States."


def test_relational_subject_is_preserved_before_final_hop_attribute() -> None:
    first_text, _ = summarize_evidence(
        {
            "title": "Alfred Maynard",
            "sentences": [
                "Alfred Maynard (1851–1916) was an English cricketer and the son of "
                "William Maynard."
            ],
        },
        "support-0",
        "When did Alfred Maynard's father die?",
    )
    second_text, _ = summarize_evidence(
        {
            "title": "William Maynard",
            "sentences": [
                "William John Maynard (18 March 1853 – 2 September 1921) was an English footballer."
            ],
        },
        "support-1",
        "When did Alfred Maynard's father die?",
    )
    assert first_text == "Alfred Maynard — father: William Maynard."
    assert second_text == "William Maynard — died: 2 September 1921."


def test_relational_subject_does_not_project_its_own_birth_date() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Raine Spencer, Countess Spencer",
            "sentences": [
                "Raine Spencer, Countess Spencer (9 September 1929 – 21 October 2016) "
                "was the daughter of Alexander McCorquodale and the romantic novelist and "
                "socialite Dame Barbara Cartland."
            ],
        },
        "support-0",
        "When was Raine Spencer, Countess Spencer's mother born?",
    )
    assert text == "Raine Spencer, Countess Spencer — mother: Dame Barbara Cartland."


def test_parent_pair_assigns_father_and_mother_by_role() -> None:
    evidence = {
        "title": "Princess Example",
        "sentences": [
            "Princess Example was the child of Frederick VIII of Schleswig-Holstein and "
            "Louise of Sweden."
        ],
    }
    father_text, _ = summarize_evidence(
        evidence,
        "support-0",
        "Who is Princess Example's father?",
    )
    mother_text, _ = summarize_evidence(
        evidence,
        "support-0",
        "Who is Princess Example's mother?",
    )
    assert father_text == "Princess Example — father: Frederick VIII of Schleswig-Holstein."
    assert mother_text == "Princess Example — mother: Louise of Sweden."


def test_named_inlaw_subject_prefers_spouse_over_own_parent() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Khenut",
            "sentences": [
                "Khenut was a wife of King Unas and the daughter of an otherwise unknown noble."
            ],
        },
        "support-0",
        "Who is Khenut's father-in-law?",
    )
    assert text == "Khenut — spouse: King Unas."


def test_parent_pair_keeps_named_mother_after_wife_descriptor() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Albrecht, Duke of Bavaria",
            "sentences": [
                "Albrecht was the son of Crown Prince Rupprecht of Bavaria and his first wife, "
                "Duchess Marie Gabrielle in Bavaria."
            ],
        },
        "support-0",
        "Where was the place of death of Albrecht, Duke of Bavaria's mother?",
    )
    assert text == ("Albrecht, Duke of Bavaria — mother: Duchess Marie Gabrielle in Bavaria.")


def test_relation_entity_parser_handles_descriptors_initials_and_aliases() -> None:
    cases = (
        (
            "Who is the father-in-law of Princess Tuoba?",
            {
                "title": "Princess Tuoba",
                "sentences": ["Her husband was Juqu Mujian (Prince Ai)."],
            },
            "Princess Tuoba — spouse: Juqu Mujian.",
        ),
        (
            "When did Matilda Amissah-Arthur's husband die?",
            {
                "title": "Matilda Amissah-Arthur",
                "sentences": [
                    "She was married to the Vice President of Ghana, Kwesi Amissah-Arthur."
                ],
            },
            "Matilda Amissah-Arthur — spouse: Kwesi Amissah-Arthur.",
        ),
        (
            "Who is the father of the director of film Kuch Khatti Kuch Meethi?",
            {
                "title": "Rahul Rawail",
                "sentences": [
                    "He is son of film director H. S. Rawail and his son Bharat is an "
                    "upcoming director."
                ],
            },
            "Rahul Rawail — father: H. S. Rawail.",
        ),
    )
    for prompt, evidence, expected in cases:
        text, _ = summarize_evidence(evidence, "support-0", prompt)
        assert text == expected


def test_relation_entity_strips_role_prefixes_but_keeps_peerage() -> None:
    spouse_text, _ = summarize_evidence(
        {
            "title": "Renee Chenault-Fattah",
            "sentences": [
                "She is married to former U.S. Congressman Chaka Fattah of the 2nd "
                "Congressional District of Pennsylvania."
            ],
        },
        "support-0",
        "When is Renee Chenault-Fattah's husband's birthday?",
    )
    father_text, _ = summarize_evidence(
        {
            "title": "Francis I of France",
            "sentences": ["He was the son of Charles, Count of Angoulême, and Louise of Savoy."],
        },
        "support-0",
        "Who is Francis I of France's father?",
    )
    assert spouse_text == "Renee Chenault-Fattah — spouse: Chaka Fattah."
    assert father_text == "Francis I of France — father: Charles, Count of Angoulême."


def test_relation_entity_handles_suffixes_and_trailing_descriptions() -> None:
    father_text, _ = summarize_evidence(
        {
            "title": "Becky Godwin",
            "sentences": [
                "Becky Godwin was the adopted daughter and only child of the then Governor "
                "of Virginia Mills E. Godwin, Jr. (1914-1999) and Katherine Beale Godwin."
            ],
        },
        "support-0",
        "What nationality is Becky Godwin's father?",
    )
    spouse_text, _ = summarize_evidence(
        {
            "title": "Servilia (wife of Catulus)",
            "sentences": [
                "Servilia was the wife of Quintus Lutatius Catulus, the consul during 102 BC."
            ],
        },
        "support-0",
        "Where did Servilia (Wife Of Catulus)'s husband die?",
    )
    assert father_text == "Becky Godwin — father: Mills E. Godwin Jr."
    assert spouse_text == ("Servilia (wife of Catulus) — spouse: Quintus Lutatius Catulus.")


def test_subject_relation_does_not_leak_from_person_biography() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Claude Binyon",
            "sentences": [
                "Claude Binyon (October 17, 1905 – February 14, 1978) was a "
                "screenwriter and director.",
                "Throughout the 1930s, Binyon's screenplays were often directed by Wesley Ruggles.",
            ],
        },
        "support-2",
        "Which film has the director who was born later, Family Honeymoon or Confess, "
        "Doctor Corda?",
    )
    assert text == "Claude Binyon — born: October 17, 1905."


def test_subject_relation_is_allowed_for_named_media_item() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Family Honeymoon",
            "sentences": [
                "Family Honeymoon is a 1948 American comedy film directed by Claude Binyon."
            ],
        },
        "support-0",
        "Which film has the director who was born later, Family Honeymoon or Confess, "
        "Doctor Corda?",
    )
    assert text == "Family Honeymoon — director: Claude Binyon."


def test_country_prefers_subject_location_over_foreign_parent_company() -> None:
    text, _ = summarize_evidence(
        {
            "title": "City National Bank (California)",
            "sentences": [
                "City National Bank is a bank headquartered at City National Plaza in Los "
                "Angeles, California.",
                "CNB is a subsidiary of the Toronto-based Royal Bank of Canada and it is the "
                "38th largest bank in the United States.",
            ],
        },
        "support-0",
        "Are City National Bank (California) and ABC Family Worldwide located in the same country?",
    )
    assert text == "City National Bank (California) — country: United States."


def test_named_spouse_relation_accepts_spouse_wording() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Louise of Mecklenburg-Güstrow",
            "sentences": [
                "Louise of Mecklenburg-Güstrow was Queen consort of Denmark and Norway as "
                "the first spouse of the King Frederick IV of Denmark."
            ],
        },
        "support-0",
        "When did Louise of Mecklenburg-Güstrow's husband die?",
    )
    assert text == "Louise of Mecklenburg-Güstrow — spouse: King Frederick IV of Denmark."


def test_performer_relation_accepts_recording_date_before_by() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Little White Lies (1930 song)",
            "sentences": [
                "It was recorded on July 25, 1930 by Fred Waring's Pennsylvanians with "
                "vocal by Clare Hanlon and The Waring Girls.",
                "A hit version was recorded by Dick Haymes on November 3, 1947.",
            ],
        },
        "support-0",
        "What nationality is the performer of song Little White Lies (1930 Song)?",
    )
    assert text.startswith(
        "Little White Lies (1930 song) — performer: Fred Waring's Pennsylvanians"
    )


def test_parent_relation_handles_role_prefix_before_named_parent() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Chiang Ching-kuo",
            "sentences": [
                "The eldest and only biological son of former president Chiang Kai-shek, "
                "he held numerous posts in the government of the Republic of China."
            ],
        },
        "support-0",
        "When did Chiang Ching-Kuo's father die?",
    )
    assert text == "Chiang Ching-kuo — father: Chiang Kai-shek."


def test_song_composer_relation_accepts_written_by_wording() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Counting Stars",
            "sentences": [
                "Counting Stars is a song by American pop rock band OneRepublic.",
                "The song was written by lead singer Ryan Tedder, and produced by Tedder "
                "and Noel Zancanella.",
            ],
        },
        "support-0",
        "What is the award that the composer of song Counting Stars won?",
    )
    assert text == "Counting Stars — composer: Ryan Tedder."


def test_subject_relations_preserve_initialed_names() -> None:
    producer_text, _ = summarize_evidence(
        {
            "title": "Aalayamani",
            "sentences": [
                "Aalayamani is a 1962 Tamil language drama film directed by K. Shankar.",
                "The film, produced by P. S. Veerappa, had musical score by "
                "Viswanathan–Ramamoorthy.",
            ],
        },
        "support-0",
        "Which film has the producer who is older than the other, Aalayamani or The Bells Go Down?",
    )
    director_text, _ = summarize_evidence(
        {
            "title": "Sreemad Bhagavad Geetha",
            "sentences": [
                "Sreemad Bhagavad Geetha is a 1977 Indian Malayalam film, directed and "
                "produced by P. Bhaskaran."
            ],
        },
        "support-0",
        "Are both director of film Sreemad Bhagavad Geetha and director of film 72 Days "
        "from the same country?",
    )
    assert producer_text == "Aalayamani — producer: P. S. Veerappa."
    assert director_text == "Sreemad Bhagavad Geetha — director: P. Bhaskaran."


def test_nationality_uses_subject_definition_sentence() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Example Director",
            "sentences": [
                "Example Director was a French film director and screenwriter.",
                "He later worked extensively in the German film industry.",
            ],
        },
        "support-1",
        "What nationality is Example Director?",
    )
    assert text == "Example Director — nationality: French."


def test_nationality_does_not_merge_unrelated_people_modifiers() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Frédéric Chopin",
            "sentences": [
                "In late summer he was invited by Jane Stirling to visit Scotland, where he "
                "stayed near Edinburgh. He called Stirling and her friends his Scottish ladies. "
                "Later he stayed in Edinburgh with the Polish physician Adam Łyszczyński."
            ],
        },
        "read-hop-1",
        "When was the Stone of Destiny returned to the country where the same individual who "
        "invited Chopin there also gave him a loan for an apartment?",
    )
    assert "Scottish-Polish" not in text


def test_nationality_keeps_subject_identity_and_drops_later_foreign_modifier() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Example Footballer",
            "sentences": [
                "Example Footballer is a Portuguese professional footballer who later played "
                "for an Italian club."
            ],
        },
        "read-hop-1",
        "What nationality is Example Footballer?",
    )
    assert text == "Example Footballer — nationality: Portuguese."


def test_nationality_preserves_hyphenated_compound_subject_identity() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Example Director",
            "sentences": ["Example Director was a Canadian-American film director."],
        },
        "read-hop-1",
        "What nationality is Example Director?",
    )
    assert text == "Example Director — nationality: Canadian-American."


def test_nationality_accepts_subject_identity_modifiers_before_nationality() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Naresh Kumar (tennis)",
            "sentences": [
                "Naresh Kumar is a former Indian tennis player who was born in Lahore. "
                "He won the Irish Championships in 1952 and 1953."
            ],
        },
        "read-hop-1",
        "What nationality is Naresh Kumar?",
    )
    assert text == "Naresh Kumar (tennis) — nationality: Indian."


def test_nationality_accepts_born_compound_subject_identity() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Allan Dwan",
            "sentences": [
                "Allan Dwan was a pioneering Canadian-born American motion picture director, "
                "producer, and screenwriter."
            ],
        },
        "read-hop-1",
        "What nationality was Allan Dwan?",
    )
    assert text == "Allan Dwan — nationality: Canadian-American."


def test_nationality_checks_later_occurrence_after_language_label() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Cristiano Ronaldo",
            "sentences": [
                "Cristiano Ronaldo dos Santos Aveiro (European Portuguese: [sample]; born 5 "
                "February 1985) is a Portuguese professional footballer who plays for an "
                "Italian club."
            ],
        },
        "read-hop-1",
        "What are the nationality and occupation of the man with the highest number of likes "
        "on Instagram?",
    )
    assert text == (
        "Cristiano Ronaldo — nationality: Portuguese; occupation: professional footballer."
    )


def test_relation_value_stops_before_release_financing_and_starring_details() -> None:
    cases = (
        (
            "Yellowstone (film)",
            "Yellowstone is a film directed by Arthur Lubin and released by Universal Studios.",
            "Do director of film Yellowstone (Film) and director of film Mr. Theertha have "
            "the same nationality?",
            "Yellowstone (film) — director: Arthur Lubin.",
        ),
        (
            "Hit Me (film)",
            "Hit Me is a 1996 film directed by Steven Shainberg starring Elias Koteas.",
            "When was the director of film Hit Me (Film) born?",
            "Hit Me (film) — director: Steven Shainberg.",
        ),
        (
            "Inchon (film)",
            "Inchon is a film directed by Terence Young and financed by Sun Myung Moon.",
            "What is the date of birth of the director of film Inchon (Film)?",
            "Inchon (film) — director: Terence Young.",
        ),
    )
    for title, sentence, prompt, expected in cases:
        text, _ = summarize_evidence({"title": title, "sentences": [sentence]}, "support-0", prompt)
        assert text == expected


def test_location_country_ignores_style_and_foreign_parent_mentions() -> None:
    abc_text, _ = summarize_evidence(
        {
            "title": "ABC Family Worldwide",
            "sentences": [
                "ABC Family Worldwide operates the U.S. cable network Freeform.",
                "It later acquired assets of a British broadcaster.",
            ],
        },
        "support-1",
        "Are ABC Family Worldwide and City National Bank located in the same country?",
    )
    house_text, _ = summarize_evidence(
        {
            "title": "Craik-Patton House",
            "sentences": [
                "Craik-Patton House is a historic home located at Charleston, West Virginia.",
                "It was built in the Greek Revival style.",
            ],
        },
        "support-1",
        "Are Borreby Castle and Craik-Patton House located in the same country?",
    )
    assert abc_text == "ABC Family Worldwide — country: United States."
    assert house_text == "Craik-Patton House — country: United States."


def test_birth_and_death_place_patterns_handle_dates_and_aliases() -> None:
    birth_text, _ = summarize_evidence(
        {
            "title": "Valdís Óskarsdóttir",
            "sentences": ["Valdís Óskarsdóttir (born 1950 in Akureyri, Iceland) is an editor."],
        },
        "support-1",
        "Where was the director of film Country Wedding born?",
    )
    alias_birth_text, _ = summarize_evidence(
        {
            "title": "David Howard",
            "sentences": ["He was born as David Paget Davis III in Philadelphia, Pennsylvania."],
        },
        "support-1",
        "Where was the director of film The Rainbow Trail born?",
    )
    death_text, _ = summarize_evidence(
        {
            "title": "Alfred Santell",
            "sentences": ["Santell died on June 19, 1981 in Salinas, California."],
        },
        "support-1",
        "Where was the place of death of the director of film Winterset?",
    )
    assert birth_text == "Valdís Óskarsdóttir — birth place: Akureyri, Iceland."
    assert alias_birth_text == "David Howard — birth place: Philadelphia, Pennsylvania."
    assert death_text == "Alfred Santell — death place: Salinas, California."


def test_award_extractor_prefers_specific_named_award() -> None:
    aerosol_text, _ = summarize_evidence(
        {
            "title": "Aerosmith",
            "sentences": [
                "The band won numerous awards during its career.",
                "The band has scored four Grammy Awards, six American Music Awards, and ten "
                "MTV Video Music Awards.",
            ],
        },
        "support-1",
        "Which award the performer of song S.O.S. (Too Bad) earned?",
    )
    tedder_text, _ = summarize_evidence(
        {
            "title": "Ryan Tedder",
            "sentences": [
                "He is a three-time recipient of the Grammy Award for Album of the Year."
            ],
        },
        "support-1",
        "What is the award that the composer of song Counting Stars won?",
    )
    assert aerosol_text == "Aerosmith — award: four Grammy Awards."
    assert tedder_text == "Ryan Tedder — award: Grammy Award for Album of the Year."


def test_country_word_inside_entity_name_does_not_trigger_country_intent() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Valdís Óskarsdóttir",
            "sentences": ["Valdís Óskarsdóttir (born 1950 in Akureyri, Iceland) is an editor."],
        },
        "support-1",
        "Where was the director of film Country Wedding born?",
    )
    assert text == "Valdís Óskarsdóttir — birth place: Akureyri, Iceland."


def test_spouse_cleaner_stops_before_whom_clause() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Bárbara Rey",
            "sentences": [
                "In 1980 she married lion tamer Ángel Cristo whom she had two children with."
            ],
        },
        "support-0",
        "Where was the husband of Bárbara Rey born?",
    )
    assert text == "Bárbara Rey — spouse: Ángel Cristo."


def test_death_date_handles_missing_birth_side_of_lifespan() -> None:
    text, _ = summarize_evidence(
        {
            "title": "David of Trebizond",
            "sentences": [
                "David Megas Komnenos (– 1 November 1463) was the last Emperor of Trebizond."
            ],
        },
        "support-1",
        "What is the date of death of Maria of Gothia's husband?",
    )
    assert text == "David of Trebizond — died: 1 November 1463."


def test_death_place_skips_non_location_in_infancy() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Anna of Saxony",
            "sentences": [
                "Maurice's only son, Albert, died in infancy.",
                "Anna was born and died in Dresden.",
            ],
        },
        "support-1",
        "Where was the place of death of Countess Anna of Nassau's mother?",
    )
    assert text == "Anna of Saxony — death place: Dresden."


def test_grandparent_second_hop_uses_requested_parent_role() -> None:
    grandmother_text, _ = summarize_evidence(
        {
            "title": "Ptolemy I Soter",
            "sentences": ["Ptolemy was the son of Lagus and Arsinoe of Macedon."],
        },
        "support-1",
        "Who is Ptolemy II Philadelphus's paternal grandmother?",
    )
    grandfather_text, _ = summarize_evidence(
        {
            "title": "Example Mother",
            "sentences": ["Example Mother was the daughter of Father Name and Mother Name."],
        },
        "support-1",
        "Who is Example Child's maternal grandfather?",
    )
    assert grandmother_text == "Ptolemy I Soter — mother: Arsinoe of Macedon."
    assert grandfather_text == "Example Mother — father: Father Name."


def test_subject_relation_stops_before_coauthor_or_recording_clause() -> None:
    director_text, _ = summarize_evidence(
        {
            "title": "Macbeth (1971 film)",
            "sentences": [
                "Macbeth is a film directed by Roman Polanski and co-written by Polanski "
                "and Kenneth Tynan."
            ],
        },
        "support-0",
        "Are the directors of Macbeth (1971 Film) and Iron Man 3 from the same country?",
    )
    composer_text, _ = summarize_evidence(
        {
            "title": "Hully Gully (song)",
            "sentences": [
                "Hully Gully is a song written by Fred Sledge Smith and Clifford Goldsmith "
                "and recorded by The Olympics."
            ],
        },
        "support-0",
        "What nationality is the composer of song Hully Gully (Song)?",
    )
    assert director_text == "Macbeth (1971 film) — director: Roman Polanski."
    assert composer_text == (
        "Hully Gully (song) — composer: Fred Sledge Smith and Clifford Goldsmith."
    )


def test_parent_parser_accepts_son_and_successor_wording() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Ahab",
            "sentences": ["Ahab was the seventh king of Israel, the son and successor of Omri."],
        },
        "support-1",
        "Who is the paternal grandfather of Ahaziah of Israel?",
    )
    assert text == "Ahab — father: Omri."


def test_spouse_parser_accepts_longtime_companion_wording() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Maria Fitzherbert",
            "sentences": [
                "Maria Fitzherbert was a longtime companion of George IV of the United Kingdom "
                "before he became king."
            ],
        },
        "support-0",
        "What is the date of death of Maria Fitzherbert's husband?",
    )
    assert text == "Maria Fitzherbert — spouse: George IV of the United Kingdom."


def test_location_and_death_cause_trim_following_prose() -> None:
    birth_text, _ = summarize_evidence(
        {
            "title": "Andy Warhol",
            "sentences": [
                "Born and raised in Pittsburgh, Warhol initially pursued a career as an "
                "illustrator."
            ],
        },
        "support-1",
        "What is the place of birth of the director of film Tub Girls?",
    )
    cause_text, _ = summarize_evidence(
        {
            "title": "Mark Robson",
            "sentences": ["He died of a heart attack after shooting his final film."],
        },
        "support-1",
        "What is the cause of death of director of film Daddy's Gone A-Hunting?",
    )
    assert birth_text == "Andy Warhol — birth place: Pittsburgh."
    assert cause_text == "Mark Robson — death cause: a heart attack."


def test_song_composer_falls_back_to_song_by_only_without_explicit_writer() -> None:
    bowie_text, _ = summarize_evidence(
        {
            "title": "Be My Wife",
            "sentences": ["Be My Wife is a song by English musician David Bowie."],
        },
        "support-0",
        "What is the place of birth of the composer of song Be My Wife?",
    )
    counting_text, _ = summarize_evidence(
        {
            "title": "Counting Stars",
            "sentences": [
                "Counting Stars is a song by American pop rock band OneRepublic.",
                "The song was written by lead singer Ryan Tedder.",
            ],
        },
        "support-0",
        "What is the award that the composer of song Counting Stars won?",
    )
    assert bowie_text == "Be My Wife — composer: David Bowie."
    assert counting_text == "Counting Stars — composer: Ryan Tedder."


def test_release_intent_accepts_came_out_first() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Play That Song (Train song)",
            "sentences": [
                "Play That Song is a song by Train.",
                "It was released on September 29, 2016.",
            ],
        },
        "support-0",
        "Which song came out first, Nuit 17 à 52 or Play That Song (Train song)?",
    )
    assert text == "Play That Song (Train song) — release: September 29, 2016."


def test_director_relation_stops_before_stars_clause() -> None:
    text, _ = summarize_evidence(
        {
            "title": "At the Villa Rose (1920 film)",
            "sentences": [
                "The feature was directed by Maurice Elvey and stars Manora Thew and "
                "Langhorn Burton."
            ],
        },
        "support-0",
        "Which film whose director was born first, At The Villa Rose (1920 Film) or "
        "Veiled Aristocrats?",
    )
    assert text == "At the Villa Rose (1920 film) — director: Maurice Elvey."


def test_inlaw_spouse_accepts_plural_royal_wives_wording() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Isetnofret",
            "sentences": [
                "Isetnofret was one of the Great Royal Wives of Pharaoh Ramesses II and "
                "was the mother of his heir."
            ],
        },
        "support-0",
        "Who is the father-in-law of Isetnofret?",
    )
    assert text.endswith("spouse: Pharaoh Ramesses II.")


def test_birth_place_trims_following_studied_clause() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Rodrigo Duterte",
            "sentences": [
                "Rodrigo Duterte was born in Maasin, Southern Leyte, Duterte studied "
                "political science at university."
            ],
        },
        "support-1",
        "What is the place of birth of Elizabeth Zimmerman's husband?",
    )
    assert text == "Rodrigo Duterte — birth place: Maasin, Southern Leyte."


def test_award_extractor_accepts_named_award_without_award_suffix() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Mahbub Ul Alam Choudhury",
            "sentences": ["He was awarded Ekushey Padak in 2009 by the Government of Bangladesh."],
        },
        "support-1",
        "Which award the founder of university Gohira Degree College got?",
    )
    assert text == "Mahbub Ul Alam Choudhury — award: Ekushey Padak."


def test_creator_relation_and_soundtrack_composer_are_projected() -> None:
    creator_text, _ = summarize_evidence(
        {
            "title": "It's All Geek to Me",
            "sentences": [
                "It's All Geek to Me is a television program created and hosted by David Pogue."
            ],
        },
        "support-0",
        "When is the creator of It's All Geek to Me's birthday?",
    )
    composer_text, _ = summarize_evidence(
        {
            "title": "Okul (film)",
            "sentences": ["Kevin Moore provided the film's soundtrack."],
        },
        "support-0",
        "What nationality is the composer of film Okul (Film)?",
    )
    assert creator_text == "It's All Geek to Me — creator: David Pogue."
    assert composer_text == "Okul (film) — composer: Kevin Moore."


def test_relation_cleanup_handles_service_studio_and_orchestra_suffixes() -> None:
    spouse_text, _ = summarize_evidence(
        {
            "title": "Mary Cyrene Burch Breckinridge",
            "sentences": [
                "Mary Cyrene Burch Breckinridge was the wife of John C. Breckinridge and "
                "served as the Second Lady of the United States."
            ],
        },
        "support-0",
        "Who is the father-in-law of Mary Cyrene Burch Breckinridge?",
    )
    director_text, _ = summarize_evidence(
        {
            "title": "Bus Stop (1956 film)",
            "sentences": [
                "Bus Stop is a film directed by Joshua Logan for 20th Century Fox, "
                "starring Marilyn Monroe."
            ],
        },
        "support-0",
        "When is the director of film Bus Stop (1956 Film)'s birthday?",
    )
    performer_text, _ = summarize_evidence(
        {
            "title": "Solo Flight (composition)",
            "sentences": [
                "Solo Flight is a 1941 instrumental song by Benny Goodman and His Orchestra."
            ],
        },
        "support-0",
        "What is the date of death of the performer of song Solo Flight (Composition)?",
    )
    assert spouse_text == "Mary Cyrene Burch Breckinridge — spouse: John C. Breckinridge."
    assert director_text == "Bus Stop (1956 film) — director: Joshua Logan."
    assert performer_text == "Solo Flight (composition) — performer: Benny Goodman."


def test_band_nationality_ignores_member_birthplaces() -> None:
    text, _ = summarize_evidence(
        {
            "title": "The Movement (dance band)",
            "sentences": [
                "The Movement was a short-lived American techno band from Los Angeles, "
                "California consisting of Costa Rican-born AJ Mora and Canadian-born "
                "Richard Vission."
            ],
        },
        "support-1",
        "Are The Movement and Blue Rodeo from the same country?",
    )
    assert text == "The Movement (dance band) — nationality: American."


def test_death_date_and_burial_handle_lifespan_year_and_buried_on() -> None:
    death_text, _ = summarize_evidence(
        {
            "title": "Ari'imate",
            "sentences": [
                "King Ari'imate Teurura'i (1824 – 14 April 1874) was a Polynesian ruler."
            ],
        },
        "support-1",
        "When did Tehaapapa II's husband die?",
    )
    burial_text, _ = summarize_evidence(
        {
            "title": "Lulach",
            "sentences": [
                "He is believed to be buried on Saint Columba's Holy Island of Iona in or "
                "around the monastery."
            ],
        },
        "support-1",
        "Where was the place of burial of Máel Snechtai of Moray's father?",
    )
    assert death_text == "Ari'imate — died: 14 April 1874."
    assert burial_text.startswith("Lulach — burial: Saint Columba's Holy Island of Iona")


def test_death_place_trims_trailing_year() -> None:
    text, _ = summarize_evidence(
        {
            "title": "George B. Seitz",
            "sentences": ["He died in Hollywood, California in 1944."],
        },
        "support-1",
        "Where was the place of death of the director of film The Women in His Life?",
    )
    assert text == "George B. Seitz — death place: Hollywood, California."


def test_teacher_semantics_handles_lifespan_country_parent_and_codirectors() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Walter Symes",
            "sentences": ["Walter Symes (1852 – 14 October 1914) was a politician."],
        },
        "support-1",
        "Who lived longer, Bob Peart or Walter Symes?",
    )
    assert text == "Walter Symes — born: 1852; died: 14 October 1914."

    text, _ = summarize_evidence(
        {
            "title": "Andrew Lau",
            "sentences": [
                "Andrew Lau is a Hong Kong film director, producer, and cinematographer."
            ],
        },
        "support-1",
        "Which country the director of film The Wesley's Mysterious File is from?",
    )
    assert text == "Andrew Lau — country: Hong Kong."

    text, _ = summarize_evidence(
        {
            "title": "John Albert Vasa",
            "sentences": [
                "He was the son of Swedish and Polish King Sigismund III Vasa and Austrian "
                "archduchess Constance of Austria."
            ],
        },
        "support-0",
        "Where did John Albert Vasa's father die?",
    )
    assert text == "John Albert Vasa — father: Sigismund III Vasa."

    text, _ = summarize_evidence(
        {
            "title": "Seth Holt",
            "sentences": [
                "Seth Holt (1923, Palestine – 14 February 1971, London) was a British film "
                "director."
            ],
        },
        "support-1",
        "Where did the director of film Taste of Fear die?",
    )
    assert text == "Seth Holt — death place: London."

    text, _ = summarize_evidence(
        {
            "title": "Watch Your Neighbor",
            "sentences": [
                "Watch Your Neighbor is a film directed by Hampton Del Ruth and Victor Heerman."
            ],
        },
        "support-0",
        "When did the director of film Watch Your Neighbor die?",
    )
    assert text == "Watch Your Neighbor — director: Hampton Del Ruth and Victor Heerman."

    text, _ = summarize_evidence(
        {
            "title": "Kokilamma",
            "sentences": ["The film has music score provided by M. S. Viswanathan."],
        },
        "support-0",
        "What is the date of death of the composer of film Kokilamma?",
    )
    assert text == "Kokilamma — composer: M. S. Viswanathan."


def test_teacher_semantics_trims_biographical_location_tail_and_keeps_parenthetical() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Diana Ross",
            "sentences": ["Born and raised in Detroit, Michigan, Ross rose to fame as a singer."],
        },
        "support-1",
        "Where was the performer of song Missing You born?",
    )
    assert text == "Diana Ross — birth place: Detroit, Michigan."

    text, _ = summarize_evidence(
        {
            "title": "André Hunebelle",
            "sentences": [
                "He was born on 1 September 1896 in Meudon (Hauts-de-Seine), and died in Nice."
            ],
        },
        "support-1",
        "Where was the director of film Le Bossu born?",
    )
    assert text == "André Hunebelle — birth place: Meudon (Hauts-de-Seine)."


def test_teacher_semantics_handles_extended_relation_and_lifespan_phrasings() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Riccardo Freda",
            "sentences": [
                "Riccardo Freda (Alexandria, Egypt, 24 February 1909 – Rome, Italy, "
                "20 December 1999) was an Italian film director."
            ],
        },
        "support-1",
        "What is the place of birth of the director of film The Horrible Dr. Hichcock?",
    )
    assert text == "Riccardo Freda — birth place: Alexandria, Egypt."

    text, _ = summarize_evidence(
        {
            "title": "Florence Lee (born 1888)",
            "sentences": [
                "She was married to Canadian-American actor, director, and writer Dell Henderson."
            ],
        },
        "support-0",
        "When did Florence Lee (Born 1888)'s husband die?",
    )
    assert text == "Florence Lee (born 1888) — spouse: Dell Henderson."

    text, _ = summarize_evidence(
        {
            "title": "The King (2019 film)",
            "sentences": [
                "It is directed by David Michôd, written by Michôd and Joel Edgerton, and stars "
                "Timothée Chalamet."
            ],
        },
        "support-0",
        "Which country the director of film The King (2019 Film) is from?",
    )
    assert text == "The King (2019 film) — director: David Michôd."

    text, _ = summarize_evidence(
        {
            "title": "U. Visweswar Rao",
            "sentences": ["He has garnered two National Film Awards, and two state Nandi Awards."],
        },
        "support-1",
        "Which award the director of film Harischandrudu got?",
    )
    assert text == "U. Visweswar Rao — award: two National Film Awards."

    text, _ = summarize_evidence(
        {
            "title": "Desert Rat Scrap Book",
            "sentences": ["The publication was launched in late 1945 and ran through early 1967."],
        },
        "support-0",
        "Which magazine was founded first, Desert Rat Scrap Book or A. Magazine?",
    )
    assert text == "Desert Rat Scrap Book — established: 1945."


def test_teacher_semantics_handles_media_metadata_and_dual_performers() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Allt flyter",
            "sentences": [
                'Allt flyter (English festival title "The Swimsuit Issue") is a 2008 Swedish '
                "film directed by Måns Herngren."
            ],
        },
        "support-1",
        "Are Black And Tan (Film) and Allt Flyter from the same country?",
    )
    assert text == "Allt flyter — nationality: Swedish."

    text, _ = summarize_evidence(
        {
            "title": "Cream (Prince song)",
            "sentences": [
                "Cream is a song by Prince and The New Power Generation, from the album "
                "Diamonds and Pearls."
            ],
        },
        "support-0",
        "Why did the performer of song Cream (Prince Song) die?",
    )
    assert text == "Cream (Prince song) — performer: Prince and The New Power Generation."

    text, _ = summarize_evidence(
        {
            "title": "Mile High (song)",
            "sentences": [
                "Mile High is a song by James Blake featuring Metro Boomin and Travis Scott."
            ],
        },
        "support-0",
        "Where was the performer of song Mile High (Song) born?",
    )
    assert text == "Mile High (song) — performer: James Blake."


def test_teacher_semantics_handles_malformed_lifespan_and_parent_direction() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Lloyd Morrell",
            "sentences": [
                "James Herbert Lloyd Morrell (called Lloyd; 12 August 190728 March 1996) "
                "was a bishop."
            ],
        },
        "support-1",
        "Who lived longer, Jesús Lara Lara or Lloyd Morrell?",
    )
    assert text == "Lloyd Morrell — born: 12 August 1907; died: 28 March 1996."

    text, _ = summarize_evidence(
        {
            "title": "David Hyrum Smith",
            "sentences": [
                "The youngest son of Joseph Smith and Emma Hale Smith, he was an influential "
                "missionary."
            ],
        },
        "support-0",
        "What is the date of birth of David Hyrum Smith's father?",
    )
    assert text == "David Hyrum Smith — father: Joseph Smith."


def test_teacher_semantics_handles_assassination_place_and_media_country_scope() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Martin Luther King Jr.",
            "sentences": ["King was assassinated on April 4 in Memphis, Tennessee."],
        },
        "support-1",
        "Where did Bernice King's father die?",
    )
    assert text == "Martin Luther King Jr. — death place: Memphis, Tennessee."

    text, _ = summarize_evidence(
        {
            "title": "Pridjel Donji",
            "sentences": [
                "Pridjel Donji is a village in the municipality of Doboj, Bosnia and Herzegovina."
            ],
        },
        "support-0",
        "Are Pridjel Donji and Gerdab located in the same country?",
    )
    assert text == "Pridjel Donji — country: Bosnia and Herzegovina."


def test_teacher_semantics_handles_director_narrative_and_extended_parent_phrasings() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Roland the Mighty",
            "sentences": [
                "Roland the Mighty is a 1956 Italian film directed by Pietro Francisci. "
                "about the Battle of Roncevaux Pass in AD 778."
            ],
        },
        "support-0",
        "Are director of film Roland The Mighty and director of film Country Of The Deaf "
        "from the same country?",
    )
    assert text == "Roland the Mighty — director: Pietro Francisci."

    text, _ = summarize_evidence(
        {
            "title": "In the Year of the Pig",
            "sentences": [
                "In the Year of the Pig is an American documentary film directed by Emile de "
                "Antonio about American involvement in the Vietnam War."
            ],
        },
        "support-0",
        "When is the director of film In The Year Of The Pig's birthday?",
    )
    assert text == "In the Year of the Pig — director: Emile de Antonio."

    text, _ = summarize_evidence(
        {
            "title": "The Preview Murder Mystery",
            "sentences": [
                "The Preview Murder Mystery is a film directed by Robert Florey and shot in the "
                "Paramount studio."
            ],
        },
        "support-0",
        "Which film has the director died later, Something Always Happens or The Preview Murder "
        "Mystery?",
    )
    assert text == "The Preview Murder Mystery — director: Robert Florey."

    text, _ = summarize_evidence(
        {
            "title": "Big Time (1929 film)",
            "sentences": [
                "Big Time is a 1929 film starring Lee Tracy.",
                "Director Kenneth Hawks was Howard Hawks' brother.",
            ],
        },
        "support-1",
        "Do both directors of films Big Time (1929 film) and The Legion Like Women share the "
        "same nationality?",
    )
    assert text == "Big Time (1929 film) — director: Kenneth Hawks."

    text, _ = summarize_evidence(
        {
            "title": "Herbert Masaryk",
            "sentences": [
                "Herbert Masaryk was the son of Tomáš Masaryk, and his American-born wife, "
                "Charlotte Garrigue."
            ],
        },
        "support-0",
        "What is the date of birth of Herbert Masaryk's mother?",
    )
    assert text == "Herbert Masaryk — mother: Charlotte Garrigue."

    text, _ = summarize_evidence(
        {
            "title": "Princess Louise of Schleswig-Holstein-Sonderburg-Glücksburg",
            "sentences": [
                "Louise was the third child and second eldest daughter of Friedrich, Duke of "
                "Schleswig-Holstein-Sonderburg-Glücksburg and Princess Adelheid of "
                "Schaumburg-Lippe and a niece of Christian IX of Denmark."
            ],
        },
        "support-1",
        "Who is the maternal grandmother of Prince Wolrad Of Waldeck And Pyrmont?",
    )
    assert (
        text == "Princess Louise of Schleswig-Holstein-Sonderburg-Glücksburg — mother: Princess "
        "Adelheid of Schaumburg-Lippe."
    )

    text, _ = summarize_evidence(
        {
            "title": "Albert III, Duke of Bavaria",
            "sentences": [
                "He was born in Wolfratshausen to Ernest, Duke of Bavaria and Elisabetta "
                "Visconti, daughter of Bernabò Visconti."
            ],
        },
        "support-1",
        "Who is the mother-in-law of Agnes Bernauer?",
    )
    assert text == "Albert III, Duke of Bavaria — mother: Elisabetta Visconti."


def test_teacher_semantics_prefers_airport_location_country_over_route_country() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Hanimaadhoo International Airport",
            "sentences": [
                "Hanimaadhoo International Airport is an airport located on the island of "
                "Hanimaadhoo in Haa Dhaalu Atoll, Maldives, opened as a domestic airport.",
                "It later introduced direct flights to India.",
            ],
        },
        "support-0",
        "Are both Hanimaadhoo International Airport and Andøya Airport, Andenes located in the "
        "same country?",
    )
    assert text == "Hanimaadhoo International Airport — country: Maldives."


def test_teacher_semantics_stops_director_before_based_upon_source_author() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Mondo (film)",
            "sentences": [
                "Mondo is a 1995 French drama film written and directed by Tony Gatlif based upon "
                "the short story by J. M. G. Le Clézio."
            ],
        },
        "support-1",
        "Do director of film Kushi (2001 Film) and director of film Mondo (Film) share the same "
        "nationality?",
    )
    assert text == "Mondo (film) — director: Tony Gatlif."


def test_teacher_semantics_uses_gendered_parent_marker_over_pair_order() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Isabella Rossellini",
            "sentences": [
                "The daughter of Swedish actress Ingrid Bergman and Italian neorealist film "
                "director Roberto Rossellini, she became an actress and filmmaker."
            ],
        },
        "support-0",
        "Why did Isabella Rossellini's mother die?",
    )
    assert text == "Isabella Rossellini — mother: Ingrid Bergman."


def test_person_country_does_not_infer_origin_from_work_or_language_context() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Anant Mahadevan",
            "sentences": [
                "Anant Mahadevan is a screenwriter, actor, and director of Hindi and Marathi "
                "films and television serials in India.",
                "Having been an integral part of the Indian television serials and Hindi movies "
                "since the 1980s, he is also involved in the professional English and Hindi "
                "theatre.",
            ],
        },
        "support-1",
        "Which country the director of film Aggar (Film) is from?",
    )
    assert text == "Anant Mahadevan — country/nationality: not stated in visible evidence."


def test_teacher_semantics_handles_relation_tail_and_numeric_entity_edges() -> None:
    cases = (
        (
            {
                "title": "Gabi Novak",
                "sentences": [
                    "She was the wife of Croatian singer-songwriter Arsen Dedić, with whom she "
                    "was married from April 1973 until his death in 2015."
                ],
            },
            "What is the date of birth of Gabi Novak's husband?",
            "Gabi Novak — spouse: Arsen Dedić.",
        ),
        (
            {
                "title": "Pierrot Lunaire (film)",
                "sentences": [
                    "Written and directed by Bruce LaBruce as an adaptation of Arnold "
                    "Schoenberg's Pierrot Lunaire, the film premiered in 2014."
                ],
            },
            "When was the director of film Pierrot Lunaire (Film) born?",
            "Pierrot Lunaire (film) — director: Bruce LaBruce.",
        ),
        (
            {
                "title": "Peter Llewelyn Davies",
                "sentences": [
                    "Peter was the middle of five sons of Arthur and Sylvia Llewelyn Davies, one "
                    "of the Llewelyn Davies boys befriended by J. M. Barrie."
                ],
            },
            "Who is Peter Llewelyn Davies's father?",
            "Peter Llewelyn Davies — father: Arthur.",
        ),
        (
            {
                "title": "Nancy Massie Meadows",
                "sentences": [
                    "Nancy was the wife of former Governor of West Virginia Clarence W. Meadows "
                    "and served as that state's First Lady."
                ],
            },
            "Where did Nancy Massie Meadows's husband study at?",
            "Nancy Massie Meadows — spouse: Clarence W. Meadows.",
        ),
        (
            {
                "title": "Ghetto Qu'ran (Forgive Me)",
                "sentences": [
                    "Ghetto Qur'an is a song by 50 Cent from his unreleased debut album."
                ],
            },
            "Which country the performer of song Ghetto Qu'Ran (Forgive Me) is from?",
            "Ghetto Qu'ran (Forgive Me) — performer: 50 Cent.",
        ),
    )
    for value, prompt, expected in cases:
        text, _ = summarize_evidence(value, "support-0", prompt)
        assert text == expected


def test_teacher_semantics_handles_credit_and_role_parent_boundaries() -> None:
    cases = (
        (
            {
                "title": "Quo Vadis (1951 film)",
                "sentences": [
                    "The score is by Miklós Rózsa and the cinematography by Robert Surtees and "
                    "William V. Skall."
                ],
            },
            "What is the date of death of the composer of film Quo Vadis (1951 Film)?",
            "Quo Vadis (1951 film) — composer: Miklós Rózsa.",
        ),
        (
            {
                "title": "Mirza Nasir Ahmad",
                "sentences": [
                    "He was elected the day after the death of his predecessor and father, "
                    "Mirza Basheer-ud-Din Mahmood Ahmad."
                ],
            },
            "What is the date of birth of Mirza Nasir Ahmad's father?",
            "Mirza Nasir Ahmad — father: Mirza Basheer-ud-Din Mahmood Ahmad.",
        ),
        (
            {
                "title": "Patrick Berg",
                "sentences": [
                    "Patrick Berg is the son of former Rosenborg and Bodø/Glimt player Ørjan Berg."
                ],
            },
            "Who is Patrick Berg's paternal grandfather?",
            "Patrick Berg — father: Ørjan Berg.",
        ),
        (
            {
                "title": "Armadillo (magazine)",
                "sentences": [
                    "Armadillo is a web-based magazine founded by Mary Hoffman and managed by her "
                    "daughter Rhiannon Lassiter."
                ],
            },
            "Which country the founder of magazine Armadillo (Magazine) is from?",
            "Armadillo (magazine) — founder: Mary Hoffman.",
        ),
        (
            {
                "title": "Always True to You in My Fashion",
                "sentences": [
                    "Always True to You in My Fashion is a 1948 show tune by Cole Porter, written "
                    "for the musical Kiss Me, Kate."
                ],
            },
            "Which country the composer of song Always True To You In My Fashion is from?",
            "Always True to You in My Fashion — composer: Cole Porter.",
        ),
    )
    for value, prompt, expected in cases:
        semantic_text, _ = summarize_evidence(value, "support-0", prompt)
        assert semantic_text == expected


def test_person_country_ignores_parental_roots_and_uses_birth_country() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Patricia Conroy",
            "sentences": [
                "Patricia Conroy was born on January 30, 1964 in Montreal, Quebec, Canada.",
                "Conroy was born to a musical family influenced by her mother's Maritime country "
                "background and her father's Irish roots.",
            ],
        },
        "read-hop-1",
        "When did Newfoundland join the country the performer of Take Me With You was in?",
    )
    assert text == "Patricia Conroy — country: Canada."


def test_outer_when_does_not_turn_nested_birthplace_into_birth_date() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Patricia Conroy",
            "sentences": [
                "Patricia Conroy was born on January 30, 1964 in Montreal, Quebec, Canada.",
                "Conroy was born to a musical family influenced by her mother's Maritime country "
                "background and her father's Irish roots.",
            ],
        },
        "read-hop-1",
        "When did Newfoundland become a province of the country where This Time's "
        "performer was born?",
    )
    assert text == "Patricia Conroy — country: Canada."


def test_spouse_stops_before_following_sentence_location() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Kseniya Boguslavskaya",
            "sentences": [
                "Born in St. Petersburg, she studied art in Paris from 1911 to 1913. "
                "She returned to St. Petersburg in 1913 and married Ivan Puni. "
                "Their apartment in St Petersburg became a meeting place for avant-garde artists."
            ],
        },
        "read-hop-0",
        "In which country is the region where the Chechen Republic is located in "
        "Kseniya Boguslavskaya's husband's homeland?",
    )
    assert text == "Kseniya Boguslavskaya — spouse: Ivan Puni."


def test_location_country_prefers_explicit_guyana_over_name_language() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Noitgedacht",
            "sentences": [
                "Noitgedacht (Dutch for 'Never Thought') is one of the villages on the Island "
                "of Wakenaam, Guyana."
            ],
        },
        "read-hop-0",
        "When did the country where the village of Noitgedacht is located become a member "
        "of CARICOM?",
    )
    assert text == "Noitgedacht — country: Guyana."


def test_country_extractor_does_not_treat_us_dollar_as_united_states() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Barfi!",
            "sentences": [
                "Made on a budget of approximately ₹30 crore (US$4.3 million), Barfi! opened "
                "worldwide on 14 September 2012. The film became one of the highest-grossing "
                "Bollywood films of 2012 in India and overseas."
            ],
        },
        "read-hop-0",
        "When was statehood conferred by the country where Barfi was popular?",
    )
    assert text == "Barfi! — country: India."


def test_location_country_recognizes_singapore_after_language_labels() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Siglap Single Member Constituency",
            "sentences": [
                "Siglap Single Member Constituency (Traditional Chinese: 實乞納單選區; "
                "Simplified Chinese: 实乞纳单选区) is a defunct Single Member Constituency "
                "in the eastern area in Singapore."
            ],
        },
        "read-hop-0",
        "When did Sang Nila Utama come to the country where Siglap is located?",
    )
    assert text == "Siglap Single Member Constituency — country: Singapore."


def test_song_subject_performer_wins_over_later_single_title() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Miss O'Dell",
            "sentences": [
                '"Miss O\'Dell" is a song by English musician George Harrison, released as the '
                'B-side of his 1973 hit single "Give Me Love (Give Me Peace on Earth)".'
            ],
        },
        "read-hop-0",
        "Who did the Miss O'Dell performer write the song Something for?",
    )
    assert text == "Miss O'Dell — performer: George Harrison."


def test_nationality_ignores_incidental_colony_modifier() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Portuguese Empire",
            "sentences": [
                "Although the royal family returned to Portugal in 1821, the interlude led to "
                "a growing desire for independence amongst Brazilians. In 1822, Dom Pedro I "
                "proclaimed the independence of Brazil. Unlike the Spanish colonies of South "
                "America, Brazil's independence was achieved without significant bloodshed."
            ],
        },
        "read-hop-2",
        "What year was hazardous working conditions limited to children, in the former "
        "colonial holding governed by the country where Modicus plays?",
    )
    assert "nationality: Spanish" not in text


def test_birthplace_does_not_treat_lifespan_as_location() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Elisha R. Potter",
            "sentences": [
                "Elisha Reynolds Potter (June 20, 1811 – April 10, 1882) was a politician "
                "and jurist from Kingston, Rhode Island."
            ],
        },
        "read-hop-0",
        "In the administrative territorial entity where the place of birth of Elisha R. "
        "Potter is located, what county is a border shared with?",
    )
    assert text == "Elisha R. Potter — birth place: not stated in visible evidence."


def test_album_subject_performer_beats_incidental_recorded_by_phrase() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Sally Can't Dance",
            "sentences": [
                "Sally Can't Dance is the fourth solo studio album by American musician Lou "
                "Reed, released in August 1974 by RCA Records. It is also the first solo Lou "
                "Reed album not to feature any songs originally recorded by Reed's earlier "
                "band, the Velvet Underground."
            ],
        },
        "read-hop-0",
        "What college did the performer of Sally Can't Dance go to?",
    )
    assert text == "Sally Can't Dance — performer: Lou Reed."


def test_complex_location_country_question_ignores_incidental_person_countries() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Muammar Gaddafi",
            "sentences": [
                "Gaddafi's alleged responsibility for the Lockerbie bombing led to Libya's "
                "label as an international pariah. A hostile relationship developed with "
                "the United States and United Kingdom, resulting in the 1986 U.S. bombing "
                "of Libya."
            ],
        },
        "read-hop-0",
        "When was the country where Arraijan is located colonized by the country where a "
        "terrorist bombing Gaddafi's Libya was supposedly involved in occurred?",
    )
    assert text == "Muammar Gaddafi — country/nationality: not stated in visible evidence."


def test_location_country_does_not_fallback_to_incidental_nationality_modifiers() -> None:
    text, _ = summarize_evidence(
        {
            "title": "British Empire",
            "sentences": [
                "In 1695, the Scottish Parliament granted a charter to the Company of "
                "Scotland, which established a settlement in 1698 on the isthmus of Panama. "
                "Besieged by neighbouring Spanish colonists, the colony was abandoned two "
                "years later."
            ],
        },
        "read-hop-3",
        "When was the country where Arraijan is located colonized by the country where a "
        "terrorist bombing Gaddafi's Libya was supposedly involved in occurred?",
    )
    assert text == "British Empire — country/nationality: not stated in visible evidence."


def test_location_country_recognizes_panama() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Arraiján",
            "sentences": [
                "Arraiján is a city and corregimiento in Arraiján District, Panamá Oeste "
                "Province, Panama with a population of 41,041 as of 2010."
            ],
        },
        "read-hop-2",
        "When was the country where Arraijan is located colonized by another country?",
    )
    assert text == "Arraiján — country: Panama."


def test_composer_credit_stops_before_following_family_description() -> None:
    text, _ = summarize_evidence(
        {
            "title": "3 (2012 Tamil film)",
            "sentences": [
                "The soundtrack and film score were composed by newcomer Anirudh "
                "Ravichander, Dhanush's cousin-in-law."
            ],
        },
        "support-0",
        "What is the place of birth of the composer of film 3 (2012 Tamil Film)?",
    )
    assert text == "3 (2012 Tamil film) — composer: Anirudh Ravichander."


def test_birthplace_does_not_reuse_death_date_location() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Gustavo Alatriste",
            "sentences": [
                "Gustavo Miguel Alatriste (25 August 1922 – 22 July 2006) was a Mexican "
                "actor and director. Alatriste died on 22 July 2006 in Houston, Texas."
            ],
        },
        "support-1",
        "Where was the director of film Aquel Famoso Remington born?",
    )
    assert text == "Gustavo Alatriste — birth place: not stated in visible evidence."


def test_location_trims_following_date_and_repeated_subject_clause() -> None:
    cases = (
        (
            {
                "title": "Werner Jacobs",
                "sentences": ["He was born in Berlin on the 24 April, 1990."],
            },
            "What is the place of birth of the director of film What Is The Matter With Willi?",
            "Werner Jacobs — birth place: Berlin.",
        ),
        (
            {
                "title": "Fränzi Aufdenblatten",
                "sentences": [
                    "Born in Zermatt, Valais, Aufdenblatten made her World Cup debut in March "
                    "2000 in a giant slalom at Sestriere."
                ],
            },
            "What is the largest lake in the country where the place of birth for Franzi "
            "Aufdenblatten is located?",
            "Fränzi Aufdenblatten — birth place: Zermatt, Valais.",
        ),
    )
    for value, prompt, expected in cases:
        text, _ = summarize_evidence(value, "support-1", prompt)
        assert text == expected


def test_month_year_is_not_a_death_place() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Balu Mahendra",
            "sentences": [
                "Following poor health, Mahendra died of cardiac arrest in February 2014."
            ],
        },
        "support-1",
        "Where was the place of death of the director of film Azhiyatha Kolangal?",
    )
    assert text == "Balu Mahendra — death place: not stated in visible evidence."


def test_recorded_by_credit_stops_before_release_date() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Say Goodbye (Chris Brown song)",
            "sentences": [
                '"Say Goodbye" is a song recorded by American singer Chris Brown. '
                "Released on August 8, 2006, it became his third top-ten single."
            ],
        },
        "read-hop-0",
        "When did the performer of Say Goodbye release Freaky Friday?",
    )
    assert text == "Say Goodbye (Chris Brown song) — performer: Chris Brown."


def test_director_credit_stops_before_filmed_at_clause() -> None:
    text, _ = summarize_evidence(
        {
            "title": "More Milk, Yvette",
            "sentences": [
                "More Milk, Yvette is an avant garde film directed by Andy Warhol and filmed "
                "at The Factory."
            ],
        },
        "read-hop-0",
        "Where was the director of More Milk, Yvette born?",
    )
    assert text == "More Milk, Yvette — director: Andy Warhol."


def test_media_location_country_prefers_shoot_or_setting_over_production_nationality() -> None:
    cases = (
        (
            {
                "title": "Beasts of No Nation (film)",
                "sentences": [
                    "Beasts of No Nation is a 2015 American war drama film written and directed "
                    "by Cary Joji Fukunaga. Shot in Ghana, the film stars Idris Elba."
                ],
            },
            "Who was the first Prime Minister of the country where Beasts of No Nation was filmed?",
            "Beasts of No Nation (film) — country: Ghana.",
        ),
        (
            {
                "title": "Through the Olive Trees",
                "sentences": [
                    "Through the Olive Trees is a 1994 film directed by Iranian director Abbas "
                    "Kiarostami, set in earthquake-ravaged Northern Iran."
                ],
            },
            "What percentage lives in the country where Through the Olive Trees takes place?",
            "Through the Olive Trees — country: Iran.",
        ),
    )
    for value, prompt, expected in cases:
        text, _ = summarize_evidence(value, "read-hop-0", prompt)
        assert text == expected


def test_language_comparison_beats_incidental_nationality_words() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Armenia",
            "sentences": [
                "The ancient Greek terms Armenia and Armenians are first mentioned by "
                "Hecataeus. Xenophon, a Greek general, relates that the people spoke a "
                "language that to his ear sounded like the language of the Persians."
            ],
        },
        "read-hop-1",
        "People who speak the language Armenia resembles the most make up what percentage?",
    )
    assert text == "Armenia — language resembles: Persian."


def test_oldest_force_prefers_relevant_nationality_over_later_foreign_force() -> None:
    text, _ = summarize_evidence(
        {
            "title": "Navy",
            "sentences": [
                "The Spanish Infantería de Marina was formed in 1537, making it the oldest, "
                "current marine force in the world. The British Royal Marines combine ship-based "
                "service with commando operations."
            ],
        },
        "read-hop-0",
        "Is Spanish as popular as the language of the nation with the oldest navy in the world?",
    )
    assert text == "Navy — oldest force nationality: Spanish."
