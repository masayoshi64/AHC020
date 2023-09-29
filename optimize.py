import optuna
from subprocess import PIPE, Popen


def objective(trial):
    max_diff = trial.suggest_int("max_diff", 1, 1000)
    start_temp = trial.suggest_float("start_temp", 100000, 10000000.0)
    end_temp = trial.suggest_float("end_temp", 100, 1000)
    p = Popen(f"psytester r -t 0-9 --tester_arguments \"{max_diff} {start_temp}, {end_temp}\"", shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    prefix = "Avg Score:"
    for line in stderr.decode("utf-8").split("\n"):
        if line.startswith(prefix):
            return float(line[len(prefix):])


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="test",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=1000)
