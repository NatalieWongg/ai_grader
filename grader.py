import json
import string
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load("en_core_web_sm")

extra_words = {"not", "without", "never"}
def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop or token.text.lower() in extra_words]
    return " ".join(tokens)

def tfidf_score(mark_scheme, student_ans): 
    vectorizer = TfidfVectorizer()
    vecs = vectorizer.fit_transform([mark_scheme, student_ans])
    return cosine_similarity(vecs[0], vecs[1])[0][0]

threshold = 0.25
def grade_answer(student_ans, question):
    
    student_ans_proc = preprocess(student_ans)
    marks_obtained = 0
    point_details = []
    
    for point in question["points"]:
        mark_scheme_text = preprocess(point["text"])
        
        tfidf_sim = tfidf_score(mark_scheme_text, student_ans_proc)
        
        awarded =  tfidf_sim >= threshold
        if awarded:
            marks_obtained += point["marks"]
        
        point_details.append({
            "point_id": point["id"],
            "point_text": point["text"],
            "tfidf_score": round(tfidf_sim, 3),
            "awarded": awarded
        })
    
    marks_obtained = min(marks_obtained, question["max_marks"])
    return marks_obtained, point_details

def grade_all(mark_scheme, student_ans):
    results = []
    
    for q in mark_scheme["questions"]:
        qid = q["id"]
        ans = student_ans.get(qid, "")
        marks, details = grade_answer(ans, q)
        
        results.append({
            "question_id": qid,
            "question_text": q["text"],
            "student_ans": ans,
            "marks_awarded": marks,
            "max_marks": q["max_marks"],
            "point_details": details
        })
    
    return results


with open("mark_scheme.json") as f:
    mark_scheme = json.load(f)

with open("student_ans.json") as f:
    student_ans = json.load(f)

results = grade_all(mark_scheme, student_ans)

for r in results:
    print(f"\n{r['question_id']} | {r['marks_awarded']}/{r['max_marks']}")
    print("Student:", r["student_ans"])
    for d in r["point_details"]:
        print(f"  - {d['point_id']} ({d['point_text']}) "
                f"TF-IDF={d['tfidf_score']} "
                f"=> {'correct' if d['awarded'] else 'incorrect'}")

flat_rows = []
for r in results:
    for d in r["point_details"]:
        flat_rows.append({
            "question_id": r["question_id"],
            "student_ans": r["student_ans"],
            "marks_awarded": r["marks_awarded"],
            "max_marks": r["max_marks"],
            "point_id": d["point_id"],
            "point_text": d["point_text"],
            "tfidf_score": d["tfidf_score"],
            "awarded": d["awarded"]
        })

df = pd.DataFrame(flat_rows)
df.to_csv("grading_results.csv", index=False)
