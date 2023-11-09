from langchain.evaluation import load_evaluator
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.criteria.eval_chain import Criteria
from colorama import Fore
from enum import Enum
from datetime import datetime
import pytz
import csv
import time
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
        template = """Given the input context, which do you prefer: A or B?
            Evaluate based on the following criteria:
            {criteria}
            Reason step by step and finally, respond with either [[A]] or [[B]] on its own line.

            DATA
            ----
            input: {input}
            A: {prediction}
            B: {prediction_b}
            ---
            Reasoning:

            """, 
        input_variables = ["criteria", "input", "prediction", "prediction_b"]
)

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
    "truthfulness": "Does the writing feel honest and sincere?",
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
    Criteria.DETAIL: "Does the submission demonstrate attention to detail?",
}

list_compiled_ans = {}

with open('./app/prev_records/comparison.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=",")
    line_count = 0
    for row in csv_reader:
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

            with open('./app/prev_records/eval.csv', mode='a') as f:
                for i in custom_criteria:
                    evaluator = load_evaluator("pairwise_string", llm=llm, criteria=i, prompt=prompt_template)

                    try:
                        output = evaluator.evaluate_string_pairs(
                            prediction = str(list_compiled_ans[row["Original_Question"]].stuff_ans),
                            prediction_b = str(list_compiled_ans[row["Original_Question"]].refine_ans),
                            input = str(list_compiled_ans[row["Original_Question"]].generated_qn),
                        )

                    except ValueError:
                        print("skipped criteria " + i)
                        time.sleep(20)
                        continue

                    else:
                        for j in output:
                            print(Fore.BLUE + f"" + j + "\n" + str(output[j]))
                        print(Fore.BLUE + f"" + "criteria: " + i)
                        
                        now = datetime.strftime(datetime.now(pytz.timezone('Asia/Singapore')), "%Y-%m-%d %H:%M:%S")
                        header = ["Time_Enquired", "Generated_Question", "A", "B", "Difference", "Reasoning", "Better_Model", "Score", "Criteria"]
                        data = [now, row['Generated_Question'], str(list_compiled_ans[row["Original_Question"]].stuff_ans), str(list_compiled_ans[row["Original_Question"]].refine_ans), "STUFF/REFINE", output['reasoning'], output['value'], str(output['score']), i]
                        writer = csv.writer(f)
                        # writer.writerow(header)
                        writer.writerow(data)
                    
                    time.sleep(20)