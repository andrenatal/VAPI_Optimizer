for each optimization iteration: 1. DSPy proposes new system prompt 2. PATCH /assistant/{A_id} → update dental assistant's prompt 3. For each test scenario (e.g. 5-8 scenarios):
a. PATCH /assistant/{B_id} → update caller persona/script
b. POST /call → Assistant B calls Assistant A's number
c. Poll GET /call/{id} until status = "ended"
d. Extract: analysis.successEvaluation,
analysis.structuredData,
artifact.transcript 4. Aggregate scores → return to DSPy metric function 5. DSPy decides next candidate
