import jax
import jax.numpy as jnp
import mlflow
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ffbq.helpers import grid, regrid
from ffbq.bq import BQ
from solstice.kernels import RFF
import numpy as np
from tensorflow_probability.substrates.jax import distributions as tfd


def generate_data(genz_problem, N, val_size, test_size, noise=None, key=jax.random.PRNGKey(2023)):
    seed = key.tolist()[1]
    X, y, y_noisy = X, y, y_noisy = genz_problem.sample(N, noise=noise, key=key)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_noisy, test_size=test_size, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def matrix_norm(aprox_mat, true_mat):
    return jnp.linalg.norm(aprox_mat - true_mat) / jnp.linalg.norm(true_mat)


def variance_metrics(fint_mu, fint_var, sol):
    sd = jnp.sqrt(fint_var)

    # various calibration metrics
    z_score = (sol - fint_mu) / sd
    one_sd = jnp.where(jnp.abs(z_score) <= 1, 1, 0)
    two_sd = jnp.where(jnp.abs(z_score) <= 2, 1, 0)

    return z_score, one_sd, two_sd


def metric_gp_bq(gp, bq_res, bq_var,
                 y_val, X_test, y_test, sol, 
                 experiment, gptype, ktype, bqtype, 
                 epochs, lr,  key, **kwargs):
    
    mlflow.set_experiment(experiment)

    N_train, d = gp.X.shape
    N_test = X_test.shape[0]
    
    # gp predictions
    gp_preds, gp_var = gp(y_val, X_test)
    gp_nll = gp.nll(y_val)

    # gp errors
    # gp_mse = mean_squared_error(y_test, gp_preds)

    # bq
    bq_err = jnp.abs(sol[0] - bq_res)
    rel_err = bq_err / jnp.abs(sol[0])

    # variance prediction
    z_score, one_sd, two_sd = variance_metrics(bq_res, bq_var, sol[0])

    with mlflow.start_run() as run:
        mlflow.set_tag("gptype", gptype)
        mlflow.set_tag("ktype", ktype)
        mlflow.set_tag("operator", bqtype)
        mlflow.log_param("sol", sol[0])
        mlflow.log_param("sol_err", sol[1])

        # problem setting parameters
        mlflow.log_param("N", N_train)
        mlflow.log_param("d", d)
        mlflow.log_param("seed", key.tolist()[1])
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)
        if bqtype in ["mc", "qmc", "fft", "rff", "ssbq", "ffbq", "lrfft"]:
            mlflow.log_param("Nconv", kwargs.get("Nconv"))

        # gp parameters
        diag = kwargs.get("diag", gp.diag)
        mlflow.log_param("diag", diag)
        mlflow.log_param("ls", jnp.exp(gp.kernel.scale))
        mlflow.log_param("var_k", gp.kernel.kernel.variance)
        R = kwargs.get("R", 0)
        mlflow.log_param("R", R)
    
        # metrics
        mlflow.log_metric("bqres", bq_res)
        mlflow.log_metric("bqvar", bq_var)
        mlflow.log_metric("nll", gp_nll)
        # mlflow.log_metric("mse", gp_mse)
        mlflow.log_metric("bqerr", bq_err)
        mlflow.log_metric("bqrel", rel_err)
        mlflow.log_metric("z_score", z_score)
        mlflow.log_metric("one_sd", one_sd)
        mlflow.log_metric("two_sd", two_sd)


def metric_gp_kmu_var(gp, bq_res, bq_var, bq_z,
                 y_val, X_test, y_test, sols, 
                 experiment, gptype, ktype, bqtype, 
                 epochs, lr,  key, **kwargs):
    
    mlflow.set_experiment(experiment)

    N_train, d = gp.X.shape
    N_test = X_test.shape[0]
    
    # gp predictions
    gp_preds, gp_var = gp(y_val, X_test)
    gp_nll = gp.nll(y_val)

    # sols
    sols_mu, sols_var, sols_z = sols

    # bq mu err
    bq_mu_err = jnp.abs(sols_mu[0] - bq_res)
    mu_rel_err = bq_mu_err / jnp.abs(sols_mu[0])

    # bq var err
    bq_var_err = jnp.abs(sols_var[0] - bq_var)
    var_rel_err = bq_var_err / jnp.abs(sols_var[0])

    # bq kmu err
    if bq_z is not None:
        bq_kmu_err = matrix_norm(bq_z, sols_z[0])
    
    with mlflow.start_run() as run:
        mlflow.set_tag("gptype", gptype)
        mlflow.set_tag("ktype", ktype)
        mlflow.set_tag("operator", bqtype)
        mlflow.log_param("sol_mu", sols_mu[0])
        mlflow.log_param("sol_var", sols_var[0])

        # problem setting parameters
        mlflow.log_param("N", N_train)
        mlflow.log_param("d", d)
        mlflow.log_param("seed", key.tolist()[1])
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)
        if bqtype in ["mc", "qmc", "fft", "rff", "ssbq", "ffbq", "lrfft"]:
            mlflow.log_param("Nconv", kwargs.get("Nconv"))

        # gp parameters
        diag = kwargs.get("diag", gp.diag)
        mlflow.log_param("diag", diag)
        mlflow.log_param("ls", jnp.exp(gp.kernel.scale))
        mlflow.log_param("var_k", gp.kernel.kernel.variance)
        R = kwargs.get("R", 0)
        mlflow.log_param("R", R)
    
        # metrics
        mlflow.log_metric("bqres", bq_res)
        mlflow.log_metric("bqvar", bq_var)
        mlflow.log_metric("nll", gp_nll)
        mlflow.log_metric("bqerr", bq_mu_err)
        mlflow.log_metric("bqrel", mu_rel_err)
        mlflow.log_metric("bqvar_err", bq_var_err)
        mlflow.log_metric("bqvar_rel", var_rel_err)
        if bq_z is not None:
            mlflow.log_metric("bqz_err", bq_kmu_err)


def metric_gp_time(gp, bq_res, bq_var, bq_z, t,
                 y_val, X_test, y_test, sols, 
                 experiment, gptype, ktype, bqtype, 
                 epochs, lr,  key, **kwargs):
    
    mlflow.set_experiment(experiment)

    N_train, d = gp.X.shape
    N_test = X_test.shape[0]
    
    # gp predictions
    gp_preds, gp_var = gp(y_val, X_test)
    gp_nll = gp.nll(y_val)

    # sols
    sols_mu, sols_var, sols_z = sols

    # bq mu err
    if sols_mu[0] is not None:
        bq_mu_err = jnp.abs(sols_mu[0] - bq_res)
        mu_rel_err = bq_mu_err / jnp.abs(sols_mu[0])

    # bq var err
    if sols_var[0] is not None:
        bq_var_err = jnp.abs(sols_var[0] - bq_var)
        var_rel_err = bq_var_err / jnp.abs(sols_var[0])

    # bq kmu err
    if bq_z is not None:
        bq_kmu_err = matrix_norm(bq_z, sols_z[0])
    
    with mlflow.start_run() as run:
        mlflow.set_tag("gptype", gptype)
        mlflow.set_tag("ktype", ktype)
        mlflow.set_tag("operator", bqtype)
        mlflow.log_param("sol_mu", sols_mu[0])
        mlflow.log_param("sol_var", sols_var[0])

        # problem setting parameters
        mlflow.log_param("N", N_train)
        mlflow.log_param("d", d)
        mlflow.log_param("seed", key.tolist()[1])
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)
        if bqtype in ["mc", "qmc", "fft", "rff", "ssbq", "ffbq", "lrfft"]:
            mlflow.log_param("Nconv", kwargs.get("Nconv"))

        # gp parameters
        diag = kwargs.get("diag", gp.diag)
        mlflow.log_param("diag", diag)
        mlflow.log_param("ls", jnp.exp(gp.kernel.scale))
        mlflow.log_param("var_k", gp.kernel.kernel.variance)
        R = kwargs.get("R", 0)
        mlflow.log_param("R", R)
    
        # metrics
        mlflow.log_metric("bqres", bq_res)
        mlflow.log_metric("bqvar", bq_var)
        mlflow.log_metric("nll", gp_nll)
        t_baseline = kwargs.get("t_baseline", 0)
        mlflow.log_metric("time", t)
        mlflow.log_metric("time_baseline", t_baseline)
        if sols_mu[0] is not None:
            mlflow.log_metric("bqerr", bq_mu_err)
            mlflow.log_metric("bqrel", mu_rel_err)
        if sols_var[0] is not None:
            mlflow.log_metric("bqvar_err", bq_var_err)
            mlflow.log_metric("bqvar_rel", var_rel_err)
        if bq_z is not None:
            mlflow.log_metric("bqz_err", bq_kmu_err)


def get_experiments(exp_name, all_cols=False):
    client = mlflow.MlflowClient()
    exp_id = client.get_experiment_by_name(exp_name).experiment_id
    runs = mlflow.search_runs(exp_id)
    
    clean_cols = [col.split(".")[-1] if "." in col else col for col in runs.columns]
    runs.columns = clean_cols

    if all_cols:
        return runs, clean_cols

    # grouping columns
    model_cols = [  # columns that are constant across seeds
        "gptype", "ktype", "operator", "d", "N", "Nconv", "R",
        "epochs"
    ]
    if "sol" not in runs.columns:
        sol_cols = ["sol_mu", "sol_var"]
    else:
        sol_cols = ["sol", "sol_err"]

    run_cols = [  # columns that will change across seeds/runs
        "run_id", "seed", "ls", "diag", "var_k", *sol_cols
    ]
    metric_cols = [  # metrics for each seed
        "bqres", "bqerr", "bqrel", "nll", #"mse",
    ]

    if "bqvar" in runs.columns:
        metric_cols += ["bqvar", "z_score", "one_sd", "two_sd"]

    if "time" in runs.columns:
        metric_cols += ["time"]

    include_cols = model_cols + run_cols + metric_cols
    cols = {"model": model_cols, "run": run_cols, "metrics": metric_cols}

    return runs[include_cols], cols


def seed_summary(exp_df, exp_cols, setting):
    # aggregate
    res_sum = exp_df.groupby(["gptype", "ktype", "operator", "sol", setting]).agg(
        {col: ["mean", "std"] for col in exp_cols["metrics"]}
    ).reset_index().sort_values(["d", ("bqerr", "mean")])
    res_sum.columns = ["_".join(col) if '' not in col else col[0] for col in res_sum.columns]

    # rel err mean and std
    res_sum["bqrel_mean"] = res_sum["bqerr_mean"] / abs(res_sum["sol"].astype(float))
    res_sum["bqrel_std"] = res_sum["bqerr_std"] / abs(res_sum["sol"].astype(float))

    # rel err mean and std
    res_sum = res_sum.sort_values(["d", "bqrel_mean"]).reset_index(drop=True)
    return res_sum


def best_operator_models(res_df, metric="bqrel_mean", setting=None):
    if setting is not None:
        best_idx = res_df.groupby([setting, "operator"])[metric].idxmin().values
        return res_df.loc[best_idx].sort_values([setting, "bqrel_mean"])
    else:
        best_idx = res_df.groupby("operator")[metric].idxmin().values
        return res_df.loc[best_idx].sort_values("bqrel_mean")


def best_operator_plurality(exp_seeds_agged, best_each_d):
    best_over_d_models = best_each_d.groupby(["operator", "ktype"]).count().reset_index().sort_values(
        ["operator" ,"gptype"]
    )[["operator", "ktype", "gptype"]].reset_index(drop=True)
    best_over_d_models = best_over_d_models.loc[best_over_d_models.groupby(
        "operator"
    )["gptype"].idxmax(), :]
    best_over_d_models = np.array(best_over_d_models[["operator", "ktype"]])
    
    # subset original data
    best_idx = []
    for i in range(best_over_d_models.shape[0]):
        best_model_idx = (
            exp_seeds_agged["operator"] == best_over_d_models[i, 0]
        ) & (
            exp_seeds_agged["ktype"] == best_over_d_models[i, 1]
        )
        best_model_idx = np.argwhere(best_model_idx).reshape(-1).tolist()
        best_idx.extend(best_model_idx)

    best_plurality_df = exp_seeds_agged.loc[best_idx].sort_values(["d", "bqrel_mean"]).reset_index(drop=True)
    return best_plurality_df
