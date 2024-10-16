from datetime import datetime
import json
import os

from src.scrub import scrub_web
from src.key3 import create_queries
from src.evaluate import sort_results, CLS_POOLING, MEAN_POOLING, MAX_POOLING

def hin_fetch(subjects, weights, pooling):
    current_time = datetime.now().strftime("%m-%d_%H-%M")
    file_path = f"logs/run_{current_time}_{weights}{pooling}.md"

    results = []

    for subject in subjects :
        log_content = f"# Hin run, {current_time}\n\nSubject : {subject}\n\n"
        
        data_path = f"web_data/{hash(subject)}.json"
    
        if os.path.exists(data_path) :
            log_content += f"## Query results from {data_path}*\n\n"
            print(f"* Subject known from {data_path}")
            
            with open(data_path, 'r', encoding='utf-8') as f:
                subject_results = json.load(f)
        else :
            queries, keyword_log = create_queries(subject)
            log_content += keyword_log

            subject_results, scrub_log = scrub_web(queries)
            log_content += scrub_log

            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(subject_results, f, ensure_ascii=False, indent=4)

            log_content += f"*Stored results in {data_path}*\n\n"
            print(f"\n* Stored results in {data_path}")

        results += subject_results

    print("N results :", len(results))

    full_scored_results = []
        
    for subject in subjects :
        print("- Subject", subject)
        scored_results, results_log = sort_results(subject, results, weights, pooling)
        log_content += results_log
        
        for result in scored_results :
            for full_result in full_scored_results:
                if full_result['title'] == result['title']:
                    full_result['score'] += result['score']
                    break
            else :
                full_scored_results.append(result)

    sorted_results = sorted(full_scored_results, key=lambda x: x['score'], reverse=True)

    print("### Done ###\n")

    report = "## Results\n"
    # Print the top 10 results
    for idx, result in enumerate(sorted_results[:10], 1):
        report += f"\nRank {idx} ({result['score']:.4f}):\nTitle: {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}\n" + "-" * 40

    print(report + "\n")

    # Create and save the file
    with open(file_path, 'w') as file:
        file.write(log_content + report)

#subject = input("Enter subject : ")
#subject = "State of the art on the identification of wood structure natural frequencies. Influence of the mechanical properties and interest in sensitivity analysis as prospects for reverse identification method of wood elastic properties."
#subject = "Experiments, numerical models and optimization of carbon-epoxy plates damped by a frequency-dependent interleaved viscoelastic layer"
#subject = "Dynamic response of carbon-epoxy laminates including a perforated viscoelastic film"
#subject = "tig welding of inconel 625 and influences on micro structures"
#subject = "Modelisation of thermo-mechanical impact of temperature oscillations on bonding and metallization for SiC MOSFETs soldered on ceramic substrate using ANSYS"
#subject = "Thermo-Mechanical Impact of temperature oscillations on bonding and metallization for SiC MOSFETs soldered on ceramic substrate"
subjects = [
    "Impact response of carbon-epoxy plates damped by a interleaved perforated viscoelastic layer",
    "Transient response of carbon-epoxy plates damped by a interleaved perforated viscoelastic layer",
    "Shock response of carbon-epoxy plates damped by a interleaved perforated viscoelastic layer",
    "Fracture of composite laminate with an inserted perforated viscoelastic layer",
    "Impact response of composite laminate damped by a interleaved perforated viscoelastic layer",
]


# hin_fetch(subject, [title_weight, snippet_weight], [title_pooling, snippet_pooling])
hin_fetch(subjects, [2,1], [CLS_POOLING, MAX_POOLING])
