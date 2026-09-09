"""
Prompt text extracted verbatim from AI4Bharat's OIWER repository
(MIT licensed, Copyright (c) 2025 AI4Bharat). Only the shared task
prompts and the Telugu guideline block are kept here.

Source: https://github.com/AI4Bharat/OIWER-Orthographically-Informed-Benchmarking-for-ASR
"""

TASK_OVERVIEW_PROMPT = '''
Imagine you are a linguist working on the {language} language. You are given the task of identifying Inverse Text normalzation variations and orthographic and normalized variations of words and expressions found in transcribed audio.

Spoken language often exhibits variability due to pronunciation, regional accents, and ambiguity in word boundaries. These variations can lead to multiple acceptable orthographic (written) and inverse text normalized (ITN) forms.

Inverse Text Normalization (ITN) refers to converting spoken or phonetically transcribed content into its standard written form. For example, “पच्चीस डॉलर” (in Hindi) can be normalized as:

✅ $25  
✅ 25 dollars  
✅ twenty-five dollars  
पच्चीस डॉलर, पच्चीस dollar, 25 डॉलर (fully native, fully converted and a mix of the conversions) are all valid variations of the spoken form.

Therefore, if the model predicts a semantically valid output that differs orthographically or in ITN formatting from the reference, it should not be unfairly penalized. Evaluation should consider such acceptable variations as correct.

To address this, your task is to enumerate all plausible written forms that convey the same meaning or intent. These variations fall into two broad types:

Important Normalization Guidelines for All Categories

  1. Atomic Unit Handling
    - Always treat complete semantic units as a whole.
    - Never split expressions like "दो करोड़", "1 जनवरी 2024", or "₹/kg" to generate partial variations.
    - Do not duplicate variations by generating variations for the sub-parts again.
    - If a multi-word or structured atomic unit has already been normalized (like a full phone number, date, address, or expression), avoid generating individual variations for internal fragments such as digits or individual words.
    - For example, if the variation group ["3512-3456-7890-123", "३५१२-३४५६-७८९०-१२३"] is defined, do not generate ["तीन", "पांच", "एक"] etc. separately again within that context.
    - This ensures **no repetition of atomic units' subcomponents**, avoiding noisy or incorrect expansion.

  2. Number and Script Variations
    - Include digit and word-based forms: "दो लाख" → ["2 लाख", "दो lakh", "2 lakh"]
    - Include comma-separated and hyphenated versions: "200000", "2,00,000", "2-लाख", "दो-lakh"
    All combinations such as full native, full digits or numeric and mixed forms should be included.

  3. Permutations and Combinations
    - For numeric + unit expressions, generate all meaningful combinations:
      e.g., "दो करोड़" → ["2 करोड़", "दो crore", "2 crore", "दो करोड़"]
    - Apply combinations for currency, measurement, time, address, etc.
    - For expressions like "₹ प्रति किलो", create: ["₹/kg", "Rs per kg", "₹ per kilogram", "rupees/kg", "INR/kg"]
    You should also include variations that contains singular and plural forms, e.g., "₹ per kg" and "₹ per kgs", rupees per kg and rupees per kgs and all combinations of these variations.

  4. Capitalization and Casing Variants
    - Always include full casing permutations:
      - "House Number" → ["House Number", "house number", "HOUSE NUMBER", "House number", "house NUMBER", "house Number"]
      - "4 PM" → ["4 PM", "4Pm", "4pm", "4 p.m.", "4 P.M.", "04:00 PM", "04 PM"]
    - Apply this to all categories

  5. Symbol, Abbreviation & Hyphenation Variants
    - Cover all known forms, such as:
      - "₹ per kg" → ["₹/kg", "Rs/kg", "₹ per kilogram", "Rs-per-kg", "Rsperkg"]
      - "kmph" → ["km/h", "km/hr", "km per hour", "kilometers per hour", "kph"]
    - Include space, slash, hyphen, and no-separator versions where applicable.

  6. Contextual Word Order Variants
    - Include alternate phrasings:
      - "4 बजे शाम" → ["4 PM", "4 pm", "evening 4 o'clock", "04:00 PM"]
      - "1 जनवरी 2024" → ["January 1, 2024", "1st January 2024", "01/01/2024", "2024-01-01"]

  7. Redundancy Avoidance
    - Do not list internal fragments of structured expressions unless needed.
      e.g., Do not extract just "₹" from "₹/kg" or "House" from "House Number 12"

  8. Apply Across All Categories
    - These rules must be applied to each of the following:
      Cardinal numbers
        → Handle all integer-based expressions with digit/word variations and unit integration.
      Currency
        → Include Rs, ₹, INR, word/digit combinations, and relevant unit casing/spelling variants.
      Mathematics
        → Represent expressions like "पांच गुना दो", "3 times 5", "5 multiplied by 3", etc., with full structure.
      Loan words and code mixing
        → Support code-mixed expressions like "Address नंबर", "Post Office", "Internet Speed" in mixed scripts.
      Numeric/Alphanumeric
        → Cover formats like PIN codes, license numbers, IDs with letter-number mix and case variants.
      Website/Email/IP
        → Include lowercase, uppercase (if spoken emphatically), and remove/add www/http where logical.
      Names and Proper Nouns
        → Maintain correct capitalization across all permutations, e.g., "India", "INDIA", "india".
      Date and Time
        → Convert to multiple formats: full date, ISO, DD-MM-YYYY, spoken variations, 24h vs 12h clock.
      Ordinal numbers
        → Include both digit and word: "तीसरा" → ["3rd", "third"], "21वां" → ["21st", "twenty-first"]
      Decimal numbers
        → Support forms like "3.14", "तीन दशमलव एक चार", "three point one four", etc.
      Abbreviations and Acronyms
        → Generate spoken + written forms: "यूएन" → ["UN", "United Nations"], with casing variants ("U.N.", "un")

    9. Symbols like -,_, space can be considered equivalent in certain contexts. So generate all combinations of these variations like non-stop-eating, non_stop_eating, non stop eating, non-stop_eating, non-stop_eating, non-stop-eating, non_stop-eating, non_stop-eating, non stop-eating, non-stop eating, non stop eating, non_stop eating.

🔹 Note:
Always keep the actual word itself as one of the variations, even if it is not the most common or expected form. This ensures that the model's output is not penalized for using a less common but still valid form and it's important to capture the full range of possible variations for the native word too.


A. Inverse Text Normalization (ITN) Guidelines  
(Where normalized forms are expected in standard written English, using only English characters, digits, and symbols)
1. Cardinal numbers  
Covers whole numbers, including ordinals, decimals, and fractions — particularly multi-word spoken forms that collectively indicate a single numeric concept.  
    Identify and group contiguous spoken tokens that together represent a number and treat them as a single unit  
    Generate variations only for the full numeric expression, not its partial subcomponents  
    Capture valid digit-based, word-based, and hybrid forms, preserving the semantic quantity they express  

2. Currency  
Covers variations in currency symbols, abbreviations, and word ordering.  
    Include symbol-based (₹, $, etc.), abbreviation-based (Rs., USD), and hybrid forms (INR 10, 10₹)  

3. Mathematics  
Covers percentages, powers, measurements, and unit-based values.  
    Include usage of symbols like %, ^2, /, -, etc.  
    Normalize to all acceptable symbolic formats (e.g., 70%, 10^2, Rs/kg)  
    Consider spelled-out or abbreviated units, with or without spacing  
    Accept all correct representations like 20kg, 20 kg, 20 kilograms  
    Allow variations such as 10-15, 10 to 15, 10–15 for ranges and lists  

4. Loan words and code mixing  
Covers spelling deviations caused by regional pronunciation and code-mixed usages.  
    Include all plausible English-script variations, even if differently pronounced (e.g., Laptop, Labtop, Laptaap)  
    

5. Numeric/Alphanumeric  
Covers phone numbers, identifiers, and other digit/letter-based formats.  
    Consider grouping and separator variations  
    Accept formats like 9876543210, 98 76 543 210, 98-76-543-210  

6. Website/Email/IP  
Covers spoken forms like “at”, “dot”, and inclusion of protocols for digital identifiers.  
    Normalize to syntactically correct formats like user@example.com, www.example.com  
    Convert spoken forms using “dot”, “at” into their symbolic equivalents  

7. Names and Proper Nouns  
Covers named entities including people, places, and organizations.  
    Do not retain fully native-script representations in the output  
    Avoid mixing native and English styles in proper nouns and treat them distinctly when found in spoken form  

8. Date and Time  
Covers multiple date formats and spoken time expressions.  
    Include all standard date formats (e.g., January 1, 2024, 01/01/2024, 2024-01-01)  
    Normalize time using either 12-hour format with AM/PM (e.g., 5:30 PM, 7 AM) or 24-hour format (e.g., 17:30, 07:00)  
    Avoid mixing native and English styles (e.g., do not retain forms like 5AM बजे)  
    Prefer standardized digital formats over spoken-style phrasing  

9. Ordinal numbers  
Included under numeric expressions but refer specifically to ordered items.  
    Normalize such expressions to standard forms like 1st, 2nd, 3rd  
    Treat these as distinct from cardinal number variants where applicable  

10. Decimal numbers  
Covers expressions with fractional numeric content.  
    Identify spoken decimals and normalize them using a dot as the decimal separator  
    Capture and preserve semantic meaning while maintaining correct notation  

11. Abbreviations and Acronyms  
Covers fully spelled or letter-by-letter spoken sequences.  
    Represent using standard uppercase forms (e.g., USA, IBM)  
    Normalize letter sequences with or without dots into consistent representations  

B. Orthographic Variation Guidelines  
(Primarily within native script)

1. Phonetic Variations  
Consider: Regional accents, dialects, or pronunciation differences.  
Action: Replace or modify letters/matras to reflect alternative pronunciations.

2. Splitting Compound Words  
Consider: Whether a compound word can be split into its constituent parts.  
Action: Decompose the word and list meaningful segments.

3. Merging Two Words  
Consider: Cases where two separate words can logically form one compound word.  
Action: Combine words to form a single, valid variation.

4. Matra and Diacritic Variations  
Consider: Alternate matras or diacritical marks (e.g., nukta in Hindi, pulli in Tamil).  
Action: Add, remove, or adjust matras and diacritics accordingly.

5. Spelling Variations for Loaned Words
Consider: Loanwords from Sanskrit, English, Persian, or other languages often have multiple recognized spellings.
Action: List all plausible forms, including transliterations and localized versions.

6. Ligature Variations  
Consider: Some scripts allow consonants to appear as ligatures or separate characters.  
Action: Provide versions both with and without ligatures.


Your output should include all valid variations per category that reflect meaning-preserving transformations. This ensures accurate, fair evaluation of spoken-to-written model outputs.
'''


TASK_INSTRUCTION_PROMPT = '''
Rules that need to be strictly followed -
1. The number of elements in the list should be equal to the number of words in the sentence, unless you decide the split the word into two words.
2. Do not generate duplicates.
3. If you have doubt whether the variation is correct or not, generate it.
4. Exhaust all possible variations of the word, do not miss anything. 
Input Format:
Language: {language}
Sentence: <the sentence will be provided here>
Output Format:
Return a list of lists containing variations, if present, for each word in the sentence.
'''


GUIDELINES_TELUGU = '''
Phonetic Variations
[
    ["ఋతువులు", "రుతువులు"],
    ["వృషభం", "రుషభం", "రిషభం"],
    ["వ్రాయడం", "రాయడం"],
    ["చేసారు", "చేశారు"],
    ["అవుతుంది", "ఔతుంది"],
    ["ఉంటుంది", "వుంటుంది"],
    ["వెళ్ళు", "వెళ్లు"],
    ["గుఱ్ఱం", "గుర్రం"],
    ["తినటం", "తినడం"],
    ["రెండవది", "రెండోది"]
]

Splitting Compound Words
[
    ["అదేవిధంగా", "అదే విధంగా"],
    ["గుర్తులేదు", "గుర్తు లేదు"],
    ["ఆడేవారు", "అడే వారు"],
    ["తయారుచేశారు", "తయారు చేశారు"],
    ["ప్రతిరోజు", "ప్రతి రోజు"],
    ["ఆంధ్రప్రదేశ్", "ఆంధ్ర ప్రదేశ్"]
]

Merging Two Words
[
    ["జీవన శైలి", "జీవనశైలి"],
    ["అదే పనిగా", "అదేపనిగా"],
    ["రైలు బండి", "రైలుబండి"],
    ["పాడి పంటలు", "పాడిపంటలు"],
    ["దేశ ముదురు", "దేశముదురు"]
]

Matra and Diacritic Variations
[
    ["ప్రతి", "ప్రతీ"],
    ["కాని", "కానీ"],
    ["ఒకొక్క", "ఒక్కొక్క"],
    ["మనం", "మనము"]
]

Spelling Variations for Loaned Words
[
    ["బిరియానీ", "బిర్యానీ"],
    ["ట్రెయిన్", "ట్రైన్"],
    ["మ్యాడమ్", "మేడమ్", "మ్యాడం"],
    ["హైదరాబాద్", "హైదరాబాదు"],
    ["ఆపిల్", "యాపిల్"],
    ["ప్లేట్", "ప్లేటు"]
]

Sandhi Rules
[
    ["వెళ్ళొస్తాను", "వెళ్ళి వస్తాను"],
    ["పన్నెండొందలు", "పన్నెండు వందలు"],
    ["వస్తుందేమో", "వస్తుంది ఏమో"],
    ["కాదంటారు", "కాదు అంటారు"],
    ["తప్పని", "తప్పు అని"],
    ["ఎన్నేళ్ళు", "ఎన్ని ఏళ్ళు"]
]
'''


