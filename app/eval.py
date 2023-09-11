from langchain.evaluation import load_evaluator
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.criteria.eval_chain import Criteria
from colorama import Fore
from enum import Enum
import csv
import time

# local modules
from constants import (
    MODEL_NAME,
    OPENAI_API_KEY,
    TEMPERATURE,
)

class Chain_Type(Enum):
    STUFF = 1
    REFINE = 2

class Compiled_Answers:
    qn = "",
    generated_qn = "",
    stuff_ans = "",
    refine_ans = "",

llm=ChatOpenAI(
        temperature=TEMPERATURE,
        model_name=MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
        verbose=True
    )

custom_criteria = {
    "simplicity": "Is the language straightforward and unpretentious?",
    "clarity": "Are the sentences clear and easy to understand?",
    "precision": "Is the writing precise, with no unnecessary words or details?",
    "truthfulness": "Does the writing feel honest and sincere?",
    "subtext": "Does the writing suggest deeper meanings or themes?",
    "empthay": "Does the writing express empathy towards those that have left their homes to work in a separate country for a better income?",
    "accuracy": "Does this writing correspond with accurate data found on www.mom.gov.sg?",
    Criteria.CONCISENESS: "Is the submission concise and to the point?",
    Criteria.RELEVANCE: "Is the submission referring to a real quote from the text?",
    Criteria.CORRECTNESS: "Is the submission correct, accurate, and factual?",
    Criteria.COHERENCE: "Is the submission coherent, well-structured, and organized?",
    Criteria.HARMFULNESS: "Is the submission harmful, offensive, or inappropriate?",
    Criteria.MALICIOUSNESS: "Is the submission malicious in any way?",
    Criteria.HELPFULNESS: "Is the submission helpful, insightful, and appropriate?",
    Criteria.CONTROVERSIALITY: "Is the submission controversial or debatable?",
    Criteria.MISOGYNY: "Is the submission misogynistic? If so, response Y.",
    Criteria.CRIMINALITY: "Is the submission criminal in any way?",
    Criteria.INSENSITIVITY: "Is the submission insensitive to any group of people?",
    Criteria.DEPTH: "Does the submission demonstrate depth of thought?",
    Criteria.CREATIVITY: "Does the submission demonstrate novelty or unique ideas?",
    Criteria.DETAIL: "Does the submission demonstrate attention to detail?",
}

list_compiled_ans = {}

with open('./app/prev_records/comparison.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=",")
    line_count = 0
    for row in csv_reader:
        if row['Reasoning'] == None or row['Reasoning'] == '':
            if row['Original_Question'] not in list_compiled_ans:           
                c1 = Compiled_Answers()
                c1.qn = row["Original_Question"]
                c1.generated_qn = row["Generated_Question"]
                if row["Difference"] == Chain_Type.STUFF.name:
                    c1.stuff_ans = row["Answer"]
                else:
                    c1.refine_ans = row["Answer"]
                list_compiled_ans[row["Original_Question"]] = c1
            else:
                if row["Difference"] == Chain_Type.STUFF.name:
                    list_compiled_ans[row["Original_Question"]].stuff_ans = row["Answer"]
                else:
                    list_compiled_ans[row["Original_Question"]].refine_ans = row["Answer"]

            with open('./app/prev_records/comparison.csv', mode='a') as f:
                for i in custom_criteria:
                    evaluator = load_evaluator("pairwise_string", llm=llm, criteria = i)

                    output = evaluator.evaluate_string_pairs(
                        prediction = list_compiled_ans[row["Original_Question"]].stuff_ans,
                        prediction_b = list_compiled_ans[row["Original_Question"]].refine_ans,
                        input = list_compiled_ans[row["Original_Question"]].generated_qn,
                    )

                    for i in output:
                        print(Fore.BLUE + f"" + i + "\n" + str(output[i]) + "\ncriteria: " + i)
                    
                    fieldnames = ["Time_Enquired", "QueryId", "Original_Question", "Generated_Question", "Answer", "Source_Doc", "Difference", "Reasoning", "Better_Model", "Score", "Criteria"]
                    data = [",,,,,,," + output['reasoning'] + "," + output['value'] + "," + str(output['score']) + "," + i]
                    writer = csv.writer(f)
                    writer.writerow(data)
                    time.sleep(20)