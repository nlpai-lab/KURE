import streamlit as st
import os
import json
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.tasks.registry import MTEB_TASKS, NANOBEIR_KO_TASK_NAMES

st.set_page_config(layout="wide")


MTEB_TASK_LIST = list(MTEB_TASKS)
NANOBEIR_TASK_LIST = list(NANOBEIR_KO_TASK_NAMES)
ALL_TASKS = MTEB_TASK_LIST + NANOBEIR_TASK_LIST


def read_score(d, score_key):
	"""Read a score from result data, handling dev/test splits."""
	scores = d.get("scores", {})
	if "dev" in scores and "test" not in scores:
		return scores["dev"][0].get(score_key)
	elif "test" in scores and "dev" not in scores:
		return scores["test"][0].get(score_key)
	elif "dev" in scores and "test" in scores:
		return scores["test"][0].get(score_key)
	return None


def app():
	data = {}
	avg_data = {}
	mteb_avg_data = {}
	nanobeir_avg_data = {}

	top_k_types = ["top10"]

	score_types = {
		"top1": ["recall_at_1", "precision_at_1", "ndcg_at_1"],
		"top3": ["recall_at_3", "precision_at_3", "ndcg_at_3"],
		"top5": ["recall_at_5", "precision_at_5", "ndcg_at_5"],
		"top10": ["recall_at_10", "precision_at_10", "ndcg_at_10"],
	}

	for task in ALL_TASKS:
		data[task] = {top_k: [] for top_k in top_k_types}

	root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

	for subdir, dirs, files in os.walk(root_dir):
		for file in files:
			for task in ALL_TASKS:
				if file == task + ".json":
					with open(os.path.join(subdir, file)) as f:
						d = json.load(f)
						for top_k in top_k_types:
							results = {}
							for score in score_types[top_k]:
								val = read_score(d, score)
								if val is None:
									break
								results[score] = val

							if len(results) != len(score_types[top_k]):
								continue

							recall = results[score_types[top_k][0]]
							precision = results[score_types[top_k][1]]
							ndcg = results[score_types[top_k][2]]
							f1_score = (
								2 * precision * recall / (precision + recall)
								if (precision + recall) > 0
								else 0
							)

							model_name = os.path.relpath(subdir, root_dir)
							data[task][top_k].append(
								(model_name, recall, precision, ndcg, f1_score)
							)

	def show_task_section(title, task_list, avg_store):
		st.markdown(f"# {title}")

		for task in task_list:
			st.markdown(f"## {task}")
			for top_k in top_k_types:
				if not data[task][top_k]:
					st.caption(f"No results for {task}")
					continue

				df = pd.DataFrame(
					data[task][top_k],
					columns=["Model", f"Recall_{top_k}", f"Precision_{top_k}", f"NDCG_{top_k}", f"F1_{top_k}"],
				)
				df = df.sort_values(by=f"NDCG_{top_k}", ascending=False)
				st.dataframe(df, use_container_width=True)

				for model_name, recall, precision, ndcg, f1 in data[task][top_k]:
					if model_name not in avg_store:
						avg_store[model_name] = {k: [[], [], [], []] for k in top_k_types}
					if model_name not in avg_data:
						avg_data[model_name] = {k: [[], [], [], []] for k in top_k_types}
					for store in [avg_store, avg_data]:
						store[model_name][top_k][0].append(recall)
						store[model_name][top_k][1].append(precision)
						store[model_name][top_k][2].append(ndcg)
						store[model_name][top_k][3].append(f1)

	def show_avg_table(title, avg_store):
		st.markdown(f"## {title}")
		for top_k in top_k_types:
			avg_results = []
			for model in avg_store:
				s = avg_store[model][top_k]
				recall_avg = sum(s[0]) / len(s[0]) if s[0] else 0
				precision_avg = sum(s[1]) / len(s[1]) if s[1] else 0
				ndcg_avg = sum(s[2]) / len(s[2]) if s[2] else 0
				f1_avg = sum(s[3]) / len(s[3]) if s[3] else 0
				avg_results.append([model, recall_avg, precision_avg, ndcg_avg, f1_avg])

			avg_df = pd.DataFrame(
				avg_results,
				columns=["Model", f"Avg Recall_{top_k}", f"Avg Precision_{top_k}", f"Avg NDCG_{top_k}", f"Avg F1_{top_k}"],
			)
			avg_df = avg_df.sort_values(by=f"Avg NDCG_{top_k}", ascending=False)
			st.dataframe(avg_df, use_container_width=True)

	# MTEB Tasks Section
	show_task_section("MTEB Korean Retrieval Tasks", MTEB_TASK_LIST, mteb_avg_data)

	# NanoBEIR-ko Tasks Section
	show_task_section("NanoBEIR-ko Tasks", NANOBEIR_TASK_LIST, nanobeir_avg_data)

	# Average Scores
	st.markdown("# Average Scores")
	show_avg_table("MTEB Average", mteb_avg_data)
	show_avg_table("NanoBEIR-ko Average", nanobeir_avg_data)
	show_avg_table("Overall Average (All 22 Tasks)", avg_data)


if __name__ == "__main__":
	app()
