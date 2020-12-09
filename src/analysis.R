#' ---
#' title: "STAT 508 Final Project Analysis"
#' author: "Callum Arnold"
#' output:
#'   html_document:
#'     code_folding: hide
#'     toc: yes
#'     toc_float: yes
#'     keep_md: true
#' ---

#+ # Set Up
# Set Up ------------------------------------------------------------------
#+ setup
knitr::opts_chunk$set(warning=FALSE, message = FALSE)

#+
library(tidyverse)
library(tidymodels)
library(baguette)
library(themis)
library(vip)
library(kknn)
library(discrim)
library(here)
library(kableExtra)
library(hrbrthemes)
library(janitor)
library(skimr)

#+
RNGkind(sample.kind = "Rounding")
set.seed(1)

#+
theme_set(theme_ipsum())

#+
credit <- as_tibble(read_csv(here("data", "creditcard.csv")))

#+
credit <- credit %>%
  mutate(
    log_amount = log(Amount + 1),
    Class = case_when(
      Class == 0 ~ "None",
      Class == 1 ~ "Fraud"
    )
  ) %>%
  mutate(across(.cols = Class, .fns = factor)) %>%
  dplyr::select(-Amount)

#+
tabyl(credit$Class)

#+
levels(credit$Class)

# Pre-Processing ----------------------------------------------------------
#+ # Pre-Processing

#' Because there's a severe class imbalance, we should use subsampling (either oversampling or undersampling).

#' Subsampling has a few important points regarding the [workflow](https://www.tidymodels.org/learn/models/sub-sampling/):

#' - It is extremely important that subsampling occurs inside of resampling. Otherwise, the resampling process can produce poor estimates of model performance.
#' - The subsampling process should only be applied to the analysis set. The assessment set should reflect the event rates seen “in the wild” and, for this reason, the skip argument to step_downsample() and other subsampling recipes steps has a default of TRUE.

#+
set.seed(1234)
credit_split <- initial_split(credit, strata = "Class", prop = 0.8)

#+
credit_train <- training(credit_split)
credit_test <- testing(credit_split)

#+
tabyl(credit_train$Class)
tabyl(credit_test$Class)

#+
# Create fold so same folds analyzed for every method
set.seed(1234)
credit_folds <- vfold_cv(credit_train, v = 10)

#+
# Create recipe so all folds are undergo same pre-processing
credit_rec <-
  recipe(Class ~ ., data = credit_train) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_zv(all_numeric()) %>%
  step_normalize(Time, log_amount) %>%
  step_downsample(Class)

#+
# Create workflow so all models can replicate same analysis methods
credit_wf <- workflow() %>%
  add_recipe(credit_rec)

#+
# Create list of metrics to record in model fits
model_mets <- metric_set(
  roc_auc, accuracy, sensitivity, specificity, j_index,
  ppv, npv, pr_auc
)

#' # Linear Models
#' ## Logistic Regression
# Logistic Regression -----------------------------------------------------

#+
# Specify logistic model
glm_spec <- logistic_reg() %>%
  set_engine("glm")

#+
# Fit logistic model to all folds in training data (resampling), saving certain metrics
# doParallel::registerDoParallel()
# glm_rs <- credit_wf %>%
#   add_model(glm_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(glm_rs, here("out", "glm_rs.rds"))
glm_rs <- readRDS(here("out", "glm_rs.rds"))

#+
# Examine which variables are most important
glm_spec %>%
  set_engine("glm") %>%
  fit(Class ~ .,
    data = juice(prep(credit_rec))
  ) %>%
  vip(geom = "point") +
  labs(title = "Logistic Regression VIP")

#+
ggsave(plot = last_plot(), path = here("out"), filename = "glm-vip.png")

#+
# Create ROC curve
glm_roc <- glm_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Logistic Regression")

#+
# Create Precision-Recall curve (PPV-Sensitivity)
glm_prc <- glm_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Logistic Regression")

#+
# Create tibble of metrics
glm_met <- glm_rs %>%
  collect_metrics() %>%
  mutate(model = "Logistic Regression")

#' ## GLMNET
# GLMNET ------------------------------------------------------------------

#+ 
# Specify GLMNET model
glmnet_spec <- logistic_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")

#+ 
# Create grid to tune penalty and mixture (Lasso vs Ridge)
glmnet_grid <- grid_latin_hypercube(
  penalty(),
  mixture(),
  size = 20
)

#+ 
# Tune GLMNET hyperparameters
# doParallel::registerDoParallel()
# set.seed(1234)
# glmnet_tune_rs <- tune_grid(
#   credit_wf %>% add_model(glmnet_spec),
#   resamples = credit_folds,
#   metrics = model_mets,
#   grid = glmnet_grid
# )
# 
# saveRDS(glmnet_tune_rs, file = here("out", "glmnet_tune_rs.rds"))
glmnet_tune_rs <- readRDS(here("out", "glmnet_tune_rs.rds"))

#+ 
# Examine AUROC for hyperparameters
glmnet_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  dplyr::select(mean, penalty, mixture) %>%
  pivot_longer(
    penalty:mixture,
    names_to = "parameter",
    values_to = "value"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point() +
  labs(
    y = "AUC",
    title = "GLMNET - AUROC vs hyperparameter tuning values",
    subtitle = "LHS grid tuning"
  ) +
  facet_wrap(~parameter, scales = "free_x")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "glmnet-auroc-tune.png")

#+ 
# Examine AUPRC for hyperparameters
glmnet_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "pr_auc") %>%
  dplyr::select(mean, penalty, mixture) %>%
  pivot_longer(
    penalty:mixture,
    names_to = "parameter",
    values_to = "value"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point() +
  labs(
    y = "AUPRC",
    title = "GLMNET - AUPRC vs hyperparameter tuning values",
    subtitle = "LHS grid tuning"
  ) +
  facet_wrap(~parameter, scales = "free_x")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "glmnet-auprc-tune.png")

#+ 
# Save best hyperparameters based on AUROC and AUPRC
best_glmnet_auroc <- select_best(glmnet_tune_rs, metric = "roc_auc")
best_glmnet_auprc <- select_best(glmnet_tune_rs, metric = "pr_auc")

#+ 
# Specify optimized GLMNET model - AUROC optimized
glmnet_final_auroc_spec <- finalize_model(
  glmnet_spec,
  best_glmnet_auroc
)

#+ 
# Specify optimized GLMNET model - AUPRC optimized
glmnet_final_auprc_spec <- finalize_model(
  glmnet_spec,
  best_glmnet_auprc
)

#+ 
# Examine which variables are most important in AUROC optimized
glmnet_final_auroc_spec %>%
  set_engine("glmnet", importance = "permutation") %>%
  fit(Class ~ .,
      data = juice(prep(credit_rec))
  ) %>%
  vip(geom = "point") +
  labs(title = "GLMNET AUROC VIP")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "glmnet-final-auroc-vip.png")

#+ 
# Examine which variables are most important in AUPRC optimized
glmnet_final_auprc_spec %>%
  set_engine("glmnet", importance = "permutation") %>%
  fit(Class ~ .,
      data = juice(prep(credit_rec))
  ) %>%
  vip(geom = "point") +
  labs(title = "GLMNET AUPRC VIP")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "glmnet-final-auprc-vip.png")

#+ 
# Fit GLMNET model to all folds in training data (resampling), saving certain metrics - AUROC optimized
# glmnet_final_auroc_rs <- credit_wf %>%
#   add_model(glmnet_final_auroc_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(glmnet_final_auroc_rs, file = here("out", "glmnet_final_auroc_rs.rds"))
glmnet_final_auroc_rs <- readRDS(here("out", "glmnet_final_auroc_rs.rds"))

#+ 
# Fit GLMNET model to all folds in training data (resampling), saving certain metrics - AUPRC optimized
# glmnet_final_auprc_rs <- credit_wf %>%
#   add_model(glmnet_final_auprc_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(glmnet_final_auprc_rs, file = here("out", "glmnet_final_auprc_rs.rds"))
glmnet_final_auprc_rs <- readRDS(here("out", "glmnet_final_auprc_rs.rds"))

#+ 
# Create ROC curve - AUROC optimized
glmnet_final_auroc_roc <- glmnet_final_auroc_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "GLMNET - AUROC")

#+ 
# Create Precision-Recall curve (PPV-Sensitivity) - AUROC optimized
glmnet_final_auroc_prc <- glmnet_final_auroc_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "GLMNET - AUROC")

#+ 
# Create tibble of metrics - AUROC optimized
glmnet_final_auroc_met <- glmnet_final_auroc_rs %>%
  collect_metrics() %>%
  mutate(model = "GLMNET - AUROC")

#+ 
# Create ROC curve - AUPRC optimized
glmnet_final_auprc_roc <- glmnet_final_auprc_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "GLMNET - AUPRC")

#+ 
# Create Precision-Recall curve (PPV-Sensitivity) - AUPRC optimized
glmnet_final_auprc_prc <- glmnet_final_auprc_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "GLMNET - AUPRC")

#+
# Create tibble of metrics - AUPRC optimized
glmnet_final_auprc_met <- glmnet_final_auprc_rs %>%
  collect_metrics() %>%
  mutate(model = "GLMNET - AUPRC")



#' # Discriminant Analysis
#' ## LDA
# LDA ---------------------------------------------------------------------

#+
# Specify LDA model
lda_spec <- discrim_linear() %>%
  set_engine("MASS")

#+
# Fit logistic model to all folds in training data (resampling), saving certain metrics
# doParallel::registerDoParallel()
# lda_rs <- credit_wf %>%
#   add_model(lda_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
#
# saveRDS(lda_rs, here("out", "lda_rs.rds"))
lda_rs <- readRDS(here("out", "lda_rs.rds"))

#+
# Create ROC curve
lda_roc <- lda_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "LDA")

#+
# Create Precision-Recall curve (PPV-Sensitivity)
lda_prc <- lda_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "LDA")

#+
# Create tibble of metrics
lda_met <- lda_rs %>%
  collect_metrics() %>%
  mutate(model = "LDA")

#' ## QDA
# QDA ---------------------------------------------------------------------
#+ 
qda_spec <- discrim_regularized(frac_common_cov = 0, frac_identity = 0) %>%
  set_engine("klaR")

#+
# Fit QDA model to all folds in training data (resampling), saving certain metrics
# doParallel::registerDoParallel()
# qda_rs <- credit_wf %>%
#   add_model(qda_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(qda_rs, here("out", "qda_rs.rds"))
qda_rs <- readRDS(here("out", "qda_rs.rds"))

#+
# Create ROC curve
qda_roc <- qda_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "QDA")

#+
# Create Precision-Recall curve (PPV-Sensitivity)
qda_prc <- qda_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "QDA")

#+
# Create tibble of metrics
qda_met <- qda_rs %>%
  collect_metrics() %>%
  mutate(model = "QDA")

#' # Tree Based Methods
#' ## Random Forest
# Random Forest -----------------------------------------------------------

#+ 
# Specify random forest model
rf_spec <- rand_forest(
  mtry = tune(),
  trees = 500,
  min_n = tune()
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

#+ 
# Tune random forest hyperparameters
# doParallel::registerDoParallel()
# set.seed(1234)
# rf_tune_rs <- tune_grid(
#   credit_wf %>% add_model(rf_spec),
#   resamples = credit_folds,
#   metrics = model_mets,
#   grid = 20
# )
# 
# saveRDS(rf_tune_rs, file = here("out", "rf_tune_rs.rds"))
rf_tune_rs <- readRDS(here("out", "rf_tune_rs.rds"))

#+ 
rf_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  dplyr::select(mean, min_n, mtry) %>%
  pivot_longer(
    cols = min_n:mtry,
    values_to = "value",
    names_to = "parameter"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(
    x = NULL, y = "AUROC",
    title = "Random Forest - AUROC vs hyperparameter tuning values",
    subtitle = "Initial tuning"
  )

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "rf-initial-auroc-tune.png")

#+ 
rf_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "pr_auc") %>%
  dplyr::select(mean, min_n, mtry) %>%
  pivot_longer(
    cols = min_n:mtry,
    values_to = "value",
    names_to = "parameter"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(
    x = NULL, y = "AUPRC",
    title = "Random Forest - AUPRC vs hyperparameter tuning values",
    subtitle = "Initial tuning"
  )

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "rf-initial-auprc-tune.png")

#' We can see that lower values of `min_n` are better, and no pattern with `mtry`
#' with respect to AUROC. For AUPRC, mtry seems optimized between 10 and 25.
#' Let's create a regular grid to do a finer optimization.

#+ 
rf_grid <- grid_regular(
  mtry(range = c(0, 25)),
  min_n(range = c(1, 10)),
  levels = 6
)

rf_grid

#+ 
# Fit Random Forest with regular tuning grid that is more focussed
# doParallel::registerDoParallel()
# set.seed(1234)
# rf_reg_tune_rs <- tune_grid(
#   credit_wf %>% add_model(rf_spec),
#   resamples = credit_folds,
#   metrics = model_mets,
#   grid = rf_grid
# )
# 
# saveRDS(rf_reg_tune_rs, file = here("out", "rf_reg_tune_rs.rds"))
rf_reg_tune_rs <- readRDS(here("out", "rf_reg_tune_rs.rds"))

#+ 
# Examine AUROC for hyperparameters
rf_reg_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(x = mtry, y = mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(
    y = "AUROC",
    title = "Random Forest - AUROC vs hyperparameter tuning values",
    subtitle = "Regular grid tuning"
  )

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "rf-grid-auroc-tune.png")

#+ 
# Examine AUPRC for hyperparameters
rf_reg_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "pr_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(x = mtry, y = mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(
    y = "AUPRC",
    title = "Random Forest - AUPRC vs hyperparameter tuning values",
    subtitle = "Regular grid tuning"
  )

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "rf-grid-auprc-tune.png")

#+ 
# Examine accuracy for hyperparameters
rf_reg_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(x = mtry, y = mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(
    y = "Accuracy",
    title = "Random Forest - Accuracy vs hyperparameter tuning values",
    subtitle = "Regular grid tuning"
  )

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "rf-grid-acc-tune.png")

#' We can see from the plot of AUROC and AUPRC that the best combination is `min_n = 1`, 
#' and `mtry = 10`. There seems to be a decline in accuracy from `mtry = 5`, however,
#' this is likely due to reduced sensitivity and improved specificity, which is
#' the opposite of what we're interested in given the class imbalance.
#' It is generally accepted that good starting points are `mtry = sqrt(p)` (c. 5)
#' and `min_n = 1` for classification models (https://bradleyboehmke.github.io/HOML/random-forest.html)

#+ 
# Save best hyperparameters based on AUROC and AUPRC
best_rf_auroc <- select_best(rf_reg_tune_rs, "roc_auc")
best_rf_auprc <- select_best(rf_reg_tune_rs, "pr_auc")

#+ 
# Specify optimized RF model - AUROC optimized
rf_final_auroc_spec <- finalize_model(
  rf_spec,
  best_rf_auroc
)

#+ 
# Specify optimized RF model - AUPRC optimized
rf_final_auprc_spec <- finalize_model(
  rf_spec,
  best_rf_auprc
)

#+ 
# Examine which variables are most important - AUROC optimized
set.seed(1234)
rf_final_auroc_spec %>%
  set_engine("ranger", importance = "permutation") %>%
  fit(Class ~ .,
    data = juice(prep(credit_rec))
  ) %>%
  vip(geom = "point") +
  labs(title = "Random Forest AUROC VIP")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "rf-final-auroc-vip.png")

#+ 
# Examine which variables are most important - AUPRC optimized
set.seed(1234)
rf_final_auprc_spec %>%
  set_engine("ranger", importance = "permutation") %>%
  fit(Class ~ .,
      data = juice(prep(credit_rec))
  ) %>%
  vip(geom = "point") +
  labs(title = "Random Forest AUPRC VIP")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "rf-final-auprc-vip.png")

#' Important to note that PCA is unsupervised so only looks at relevance to the
#' variance observed in the predictors, not at their relevance to the outcome,
#' so not necessary that PC1 would be the most important PC in predicting Class

#+ 
# Fit random forest model to all folds in training data (resampling), saving certain metrics - AUROC optimized
# rf_final_auroc_rs <- credit_wf %>%
#   add_model(rf_final_auroc_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(rf_final_auroc_rs, file = here("out", "rf_final_auroc_rs.rds"))
rf_final_auroc_rs <- readRDS(here("out", "rf_final_auroc_rs.rds"))

#+ 
# Fit random forest model to all folds in training data (resampling), saving certain metrics - AUPRC optimized
# rf_final_auprc_rs <- credit_wf %>%
#   add_model(rf_final_auprc_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(rf_final_auprc_rs, file = here("out", "rf_final_auprc_rs.rds"))
rf_final_auprc_rs <- readRDS(here("out", "rf_final_auprc_rs.rds"))

#+ 
# Create ROC curve - AUROC optimized
rf_final_auroc_roc <- rf_final_auroc_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Random Forest - AUROC")

#+ 
# Create Precision-Recall curve (PPV-Sensitivity) - AUROC optimized
rf_final_auroc_prc <- rf_final_auroc_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Random Forest - AUROC")

#+ 
# Create tibble of metrics - AUROC optimized
rf_final_auroc_met <- rf_final_auroc_rs %>%
  collect_metrics() %>%
  mutate(model = "Random Forest - AUROC")

#+ 
# Create ROC curve - AUPRC optimized
rf_final_auprc_roc <- rf_final_auprc_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Random Forest - AUPRC")

#+ 
# Create Precision-Recall curve (PPV-Sensitivity) - AUPRC optimized
rf_final_auprc_prc <- rf_final_auprc_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Random Forest - AUPRC")

#+ 
# Create tibble of metrics - AUPRC optimized
rf_final_auprc_met <- rf_final_auprc_rs %>%
  collect_metrics() %>%
  mutate(model = "Random Forest - AUPRC")

#' ## XGBoost
# XGBoost -----------------------------------------------------------------

#+ 
# Specify boosted tree model
xgb_spec <- boost_tree(
  trees = 500,
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune(),
  learn_rate = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

#+ 
# Create space filling parameter latin hypercube grid - regular grid too slow
xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), credit_train),
  learn_rate(),
  size = 20
)

#+ 
# Tune XGBoost hyperparameters using space filling parameter grid
# doParallel::registerDoParallel()
# set.seed(1234)
# xgb_tune_rs <- tune_grid(
#   credit_wf %>% add_model(xgb_spec),
#   resamples = credit_folds,
#   metrics = model_mets,
#   grid = xgb_grid
# )
# 
# saveRDS(xgb_tune_rs, file = here("out", "xgb_tune_rs.rds"))
xgb_tune_rs <- readRDS(here("out", "xgb_tune_rs.rds"))

#+ 
# Examine AUROC for hyperparameters
xgb_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  dplyr::select(mean, mtry:sample_size) %>%
  pivot_longer(
    mtry:sample_size,
    names_to = "parameter",
    values_to = "value"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point() +
  labs(
    y = "AUROC",
    title = "XGBoost - AUROC vs hyperparameter tuning values",
    subtitle = "LHS grid tuning"
  ) +
  facet_wrap(~parameter, scales = "free_x")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "xgb-auroc-tune.png")

#+ 
# Examine AUPRC for hyperparameters
xgb_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "pr_auc") %>%
  dplyr::select(mean, mtry:sample_size) %>%
  pivot_longer(
    mtry:sample_size,
    names_to = "parameter",
    values_to = "value"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point() +
  labs(
    y = "AUPRC",
    title = "XGBoost - AUPRC vs hyperparameter tuning values",
    subtitle = "LHS grid tuning"
  ) +
  facet_wrap(~parameter, scales = "free_x")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "xgb-auprc-tune.png")

#+ 
show_best(xgb_tune_rs, "roc_auc")
show_best(xgb_tune_rs, "pr_auc")

#+ 
# Save best hyperparameters based on AUROC and AUPRC
best_xgb_auroc <- select_best(xgb_tune_rs, "roc_auc")
best_xgb_auprc <- select_best(xgb_tune_rs, "pr_auc")

#+ 
# Specify optimized XGBoost model - AUROC optimized
xgb_final_auroc_spec <- finalize_model(
  xgb_spec,
  best_xgb_auroc
)

#+ 
# Specify optimized XGBoost model - AUPRC optimized
xgb_final_auprc_spec <- finalize_model(
  xgb_spec,
  best_xgb_auprc
)

#+ 
# Examine which variables are most important - AUROC optimized
set.seed(1234)
xgb_final_auroc_spec %>%
  set_engine("xgboost", importance = "permutation") %>%
  fit(Class ~ .,
    data = juice(prep(credit_rec))
  ) %>%
  vip(geom = "point") +
  labs(title = "XGBoost AUROC VIP")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "xgb-final-auroc-vip.png")

#+ 
# Examine which variables are most important - AUPRC optimized
set.seed(1234)
xgb_final_auprc_spec %>%
  set_engine("xgboost", importance = "permutation") %>%
  fit(Class ~ .,
      data = juice(prep(credit_rec))
  ) %>%
  vip(geom = "point") +
  labs(title = "XGBoost AUPRC VIP")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "xgb-final-auprc-vip.png")

#+ 
# Fit XGBoost model to all folds in training data (resampling), saving certain metrics - AUROC optimized
# xgb_final_auroc_rs <- credit_wf %>%
#   add_model(xgb_final_auroc_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(xgb_final_auroc_rs, file = here("out", "xgb_final_auroc_rs.rds"))
xgb_final_auroc_rs <- readRDS(here("out", "xgb_final_auroc_rs.rds"))

#+ 
# Fit XGBoost model to all folds in training data (resampling), saving certain metrics - AUPRC optimized
# xgb_final_auprc_rs <- credit_wf %>%
#   add_model(xgb_final_auprc_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(xgb_final_auprc_rs, file = here("out", "xgb_final_auprc_rs.rds"))
xgb_final_auprc_rs <- readRDS(here("out", "xgb_final_auprc_rs.rds"))

#+ 
# Create ROC curve - AUROC optimized
xgb_final_auroc_roc <- xgb_final_auroc_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "XGBoost - AUROC")

#+ 
# Create Precision-Recall curve (PPV-Sensitivity) - AUROC optimized
xgb_final_auroc_prc <- xgb_final_auroc_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "XGBoost - AUROC")

#+ 
# Create tibble of metrics - AUROC optimized
xgb_final_auroc_met <- xgb_final_auroc_rs %>%
  collect_metrics() %>%
  mutate(model = "XGBoost - AUROC")

#+ 
# Create ROC curve - AUPRC optimized
xgb_final_auprc_roc <- xgb_final_auprc_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "XGBoost - AUPRC")

#+ 
# Create Precision-Recall curve (PPV-Sensitivity) - AUPRC optimized
xgb_final_auprc_prc <- xgb_final_auprc_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "XGBoost - AUPRC")

#+ 
# Create tibble of metrics - AUPRC optimized
xgb_final_auprc_met <- xgb_final_auprc_rs %>%
  collect_metrics() %>%
  mutate(model = "XGBoost - AUPRC")

#' ## Bagged Tree
# Bagged Tree -------------------------------------------------------------

#+ 
# Specify bagged tree model
bag_spec <- bag_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

#+ 
# Create space filling parameter latin hypercube grid - regular grid too slow
bag_grid <- grid_latin_hypercube(
  cost_complexity(),
  tree_depth(),
  min_n(),
  size = 20
)

#+ 
# Tune bagged tree hyperparameters using space filling parameter grid
# doParallel::registerDoParallel()
# set.seed(1234)
# bag_tune_rs <- tune_grid(
#   credit_wf %>% add_model(bag_spec),
#   resamples = credit_folds,
#   metrics = model_mets,
#   grid = bag_grid
# )
# 
# saveRDS(bag_tune_rs, file = here("out", "bag_tune_rs.rds"))
bag_tune_rs <- readRDS(here("out", "bag_tune_rs.rds"))

#+ 
# Examine AUROC for hyperparameters
bag_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  dplyr::select(mean, cost_complexity:min_n) %>%
  pivot_longer(
    cost_complexity:min_n,
    names_to = "parameter",
    values_to = "value"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point() +
  labs(
    y = "AUROC",
    title = "Bagged Tree - AUROC vs hyperparameter tuning values",
    subtitle = "LHS grid tuning"
  ) +
  facet_wrap(~parameter, scales = "free_x")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "bag-auroc-tune.png")

#+ 
# Examine AUPRC for hyperparameters
bag_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "pr_auc") %>%
  dplyr::select(mean, cost_complexity:min_n) %>%
  pivot_longer(
    cost_complexity:min_n,
    names_to = "parameter",
    values_to = "value"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point() +
  labs(
    y = "AUPRC",
    title = "Bagged Tree - AUPRC vs hyperparameter tuning values",
    subtitle = "LHS grid tuning"
  ) +
  facet_wrap(~parameter, scales = "free_x")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "bag-auprc-tune.png")

#+ 
# Save best hyperparameters based on AUROC and AUPRC
best_bag_auroc <- select_best(bag_tune_rs, "roc_auc")
best_bag_auprc <- select_best(bag_tune_rs, "pr_auc")

#+ 
# Specify optimized bagged tree model - AUROC optimized
bag_final_auroc_spec <- finalize_model(
  bag_spec,
  best_bag_auroc
)

#+ 
# Specify optimized bagged tree model - AUPRC optimized
bag_final_auprc_spec <- finalize_model(
  bag_spec,
  best_bag_auprc
)

#+ 
# Examine which variables are most important - AUROC optimized
set.seed(1234)
bag_auroc_imp <- bag_final_auroc_spec %>%
  set_engine("rpart") %>%
  fit(Class ~ .,
    data = juice(prep(credit_rec))
  )

#+ 
bag_auroc_imp$fit$imp %>%
  mutate(term = fct_reorder(term, value)) %>%
  head(10) %>%
  ggplot(aes(x = value, y = term)) +
  geom_point() +
  labs(title = "Bagged Tree AUROC VIP")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "bag-final-auroc-vip.png")

#+ 
# Examine which variables are most important - AUPRC optimized
set.seed(1234)
bag_auprc_imp <- bag_final_auprc_spec %>%
  set_engine("rpart") %>%
  fit(Class ~ .,
      data = juice(prep(credit_rec))
  )

#+ 
bag_auprc_imp$fit$imp %>%
  mutate(term = fct_reorder(term, value)) %>%
  head(10) %>%
  ggplot(aes(x = value, y = term)) +
  geom_point() +
  labs(title = "Bagged Tree AUPRC VIP")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "bag-final-auprc-vip.png")

#+ 
# Fit bagged tree model to all folds in training data (resampling), saving certain metrics - AUROC optimized
# bag_final_auroc_rs <- credit_wf %>%
#   add_model(bag_final_auroc_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(bag_final_auroc_rs, file = here("out", "bag_final_auroc_rs.rds"))
bag_final_auroc_rs <- readRDS(here("out", "bag_final_auroc_rs.rds"))

#+ 
# Fit bagged tree model to all folds in training data (resampling), saving certain metrics - AUPRC optimized
# bag_final_auprc_rs <- credit_wf %>%
#   add_model(bag_final_auprc_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(bag_final_auprc_rs, file = here("out", "bag_final_auprc_rs.rds"))
bag_final_auprc_rs <- readRDS(here("out", "bag_final_auprc_rs.rds"))

#+ 
# Create ROC curve - AUROC optimized
bag_final_auroc_roc <- bag_final_auroc_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Bagged Tree - AUROC")

#+ 
# Create Precision-Recall curve (PPV-Sensitivity) - AUROC optimized
bag_final_auroc_prc <- bag_final_auroc_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Bagged Tree - AUROC")

#+ 
# Create tibble of metrics - AUROC optimized
bag_final_auroc_met <- bag_final_auroc_rs %>%
  collect_metrics() %>%
  mutate(model = "Bagged Tree - AUROC")

#+ 
# Create ROC curve - AUPRC optimized
bag_final_auprc_roc <- bag_final_auprc_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Bagged Tree - AUPRC")

#+ 
# Create Precision-Recall curve (PPV-Sensitivity) - AUPRC optimized
bag_final_auprc_prc <- bag_final_auprc_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Bagged Tree - AUPRC")

#+ 
# Create tibble of metrics - AUPRC optimized
bag_final_auprc_met <- bag_final_auprc_rs %>%
  collect_metrics() %>%
  mutate(model = "Bagged Tree - AUPRC")

#' # SVM Methods
#' ## SVM - Radial Kernel
# SVM - Radial ------------------------------------------------------------

#+
# Specify SVM-Radial model
svmr_spec <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

#+
svmr_grid <- grid_latin_hypercube(
  cost(),
  rbf_sigma(),
  size = 20
)

#+
# Tune SVM-Radial hyperparameters
# doParallel::registerDoParallel()
# set.seed(1234)
# svmr_tune_rs <- tune_grid(
#   credit_wf %>% add_model(svmr_spec),
#   resamples = credit_folds,
#   metrics = model_mets,
#   grid = svmr_grid
# )
# 
# saveRDS(svmr_tune_rs, file = here("out", "svmr_tune_rs.rds"))
svmr_tune_rs <- readRDS(here("out", "svmr_tune_rs.rds"))

#+ 
# Examine AUROC for hyperparameters
svmr_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  dplyr::select(mean, cost, rbf_sigma) %>%
  pivot_longer(
    cost:rbf_sigma,
    names_to = "parameter",
    values_to = "value"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point() +
  labs(
    y = "AUROC",
    title = "SVM Radial - AUROC vs hyperparameter tuning values",
    subtitle = "LHS grid tuning"
  ) +
  facet_wrap(~parameter, scales = "free_x")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "svmr-auroc-tune.png")

#+ 
# Examine AUPRC for hyperparameters
svmr_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "pr_auc") %>%
  dplyr::select(mean, cost, rbf_sigma) %>%
  pivot_longer(
    cost:rbf_sigma,
    names_to = "parameter",
    values_to = "value"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point() +
  labs(
    y = "AUPRC",
    title = "SVM Radial - AUPRC vs hyperparameter tuning values",
    subtitle = "LHS grid tuning"
  ) +
  facet_wrap(~parameter, scales = "free_x")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "svmr-auprc-tune.png")

#+ 
# Save best hyperparameters based on AUROC and AUPRC
best_svmr_auroc <- select_best(svmr_tune_rs, metric = "roc_auc")
best_svmr_auprc <- select_best(svmr_tune_rs, metric = "pr_auc")

#+ 
# Specify optimized SVM-R model - AUROC optimized
svmr_final_auroc_spec <- finalize_model(
  svmr_spec,
  best_svmr_auroc
)

#+ 
# Specify optimized SVM-R model - AUPRC optimized
svmr_final_auprc_spec <- finalize_model(
  svmr_spec,
  best_svmr_auprc
)

#+ 
# Fit SVM-R model to all folds in training data (resampling), saving certain metrics - AUROC optimized
# svmr_final_auroc_rs <- credit_wf %>%
#   add_model(svmr_final_auroc_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(svmr_final_auroc_rs, file = here("out", "svmr_final_auroc_rs.rds"))
svmr_final_auroc_rs <- readRDS(here("out", "svmr_final_auroc_rs.rds"))

#+ 
# Fit SVM-R model to all folds in training data (resampling), saving certain metrics - AUPRC optimized
# svmr_final_auprc_rs <- credit_wf %>%
#   add_model(svmr_final_auprc_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(svmr_final_auprc_rs, file = here("out", "svmr_final_auprc_rs.rds"))
svmr_final_auprc_rs <- readRDS(here("out", "svmr_final_auprc_rs.rds"))

#+ 
# Create ROC curve - AUROC optimized
svmr_final_auroc_roc <- svmr_final_auroc_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "SVM-R - AUROC")

#+ 
# Create Precision-Recall curve (PPV-Sensitivity) - AUROC optimized
svmr_final_auroc_prc <- svmr_final_auroc_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "SVM-R - AUROC")

#+ 
# Create tibble of metrics - AUROC optimized
svmr_final_auroc_met <- svmr_final_auroc_rs %>%
  collect_metrics() %>%
  mutate(model = "SVM-R - AUROC")

#+ 
# Create ROC curve - AUPRC optimized
svmr_final_auprc_roc <- svmr_final_auprc_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "SVM-R - AUPRC")

#+ 
# Create Precision-Recall curve (PPV-Sensitivity) - AUPRC optimized
svmr_final_auprc_prc <- svmr_final_auprc_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "SVM-R - AUPRC")

#+ 
# Create tibble of metrics - AUPRC optimized
svmr_final_auprc_met <- svmr_final_auprc_rs %>%
  collect_metrics() %>%
  mutate(model = "SVM-R - AUPRC")

#' ## SVM - Polynomial Kernel
# SVM - Polynomial --------------------------------------------------------

#+ 
# Specify SVM-Polynomial model
svmp_spec <- svm_poly(
  cost = tune(),
  degree = tune(),
  scale_factor = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

#+ 
svmp_grid <- grid_latin_hypercube(
  cost(),
  degree(),
  scale_factor(),
  size = 20
)

#+ 
# Tune SVM-P hyperparameters
# doParallel::registerDoParallel()
# set.seed(1234)
# svmp_tune_rs <- tune_grid(
#   credit_wf %>% add_model(svmp_spec),
#   resamples = credit_folds,
#   metrics = model_mets,
#   grid = svmp_grid
# )
# 
# saveRDS(svmp_tune_rs, file = here("out", "svmp_tune_rs.rds"))
svmp_tune_rs <- readRDS(here("out", "svmp_tune_rs.rds"))

#+ 
# Examine AUROC for hyperparameters
svmp_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  dplyr::select(mean, cost:scale_factor) %>%
  pivot_longer(
    cost:scale_factor,
    names_to = "parameter",
    values_to = "value"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point() +
  labs(
    y = "AUROC",
    title = "SVM Polynomial - AUROC vs hyperparameter tuning values",
    subtitle = "LHS grid tuning"
  ) +
  facet_wrap(~parameter, scales = "free_x")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "svmp-auroc-tune.png")

#+ 
# Examine AUROC for hyperparameters
svmp_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "pr_auc") %>%
  dplyr::select(mean, cost:scale_factor) %>%
  pivot_longer(
    cost:scale_factor,
    names_to = "parameter",
    values_to = "value"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point() +
  labs(
    y = "AUPRC",
    title = "SVM Polynomial - AUPRC vs hyperparameter tuning values",
    subtitle = "LHS grid tuning"
  ) +
  facet_wrap(~parameter, scales = "free_x")

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "svmp-auprc-tune.png")

#+ 
# Save best hyperparameters based on AUROC and AUPRC
best_svmp_auroc <- select_best(svmp_tune_rs, metric = "roc_auc")
best_svmp_auprc <- select_best(svmp_tune_rs, metric = "pr_auc")

#+ 
# Specify optimized SVM-P model - AUROC optimized
svmp_final_auroc_spec <- finalize_model(
  svmp_spec,
  best_svmp_auroc
)

#+ 
# Specify optimized SVM-P model - AUPRC optimized
svmp_final_auprc_spec <- finalize_model(
  svmp_spec,
  best_svmp_auprc
)

#+ 
# Fit SVM-P model to all folds in training data (resampling), saving certain metrics - AUROC optimized
# svmp_final_auroc_rs <- credit_wf %>%
#   add_model(svmp_final_auroc_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(svmp_final_auroc_rs, file = here("out", "svmp_final_auroc_rs.rds"))
svmp_final_auroc_rs <- readRDS(here("out", "svmp_final_auroc_rs.rds"))

#+ 
# Fit SVM-P model to all folds in training data (resampling), saving certain metrics - AUPRC optimized
# svmp_final_auprc_rs <- credit_wf %>%
#   add_model(svmp_final_auprc_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(svmp_final_auprc_rs, file = here("out", "svmp_final_auprc_rs.rds"))
svmp_final_auprc_rs <- readRDS(here("out", "svmp_final_auprc_rs.rds"))

#+ 
# Create ROC curve - AUROC optimized
svmp_final_auroc_roc <- svmp_final_auroc_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "SVM-P - AUROC")

#+ 
# Create Precision-Recall curve (PPV-Sensitivity) - AUROC optimized
svmp_final_auroc_prc <- svmp_final_auroc_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "SVM-P - AUROC")

#+ 
# Create tibble of metrics - AUROC optimized
svmp_final_auroc_met <- svmp_final_auroc_rs %>%
  collect_metrics() %>%
  mutate(model = "SVM-P - AUROC")

#+ 
# Create ROC curve - AUPRC optimized
svmp_final_auprc_roc <- svmp_final_auprc_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "SVM-P - AUPRC")

#+ 
# Create Precision-Recall curve (PPV-Sensitivity) - AUPRC optimized
svmp_final_auprc_prc <- svmp_final_auprc_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "SVM-P - AUPRC")

#+ 
# Create tibble of metrics - AUPRC optimized
svmp_final_auprc_met <- svmp_final_auprc_rs %>%
  collect_metrics() %>%
  mutate(model = "SVM-P - AUPRC")

#' # kNN
# kNN ---------------------------------------------------------------------

#+ 
# Specify kNN model
knn_spec <- nearest_neighbor(
  neighbors = tune()
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

#+ 
knn_grid <- grid_regular(
  neighbors(range = c(1, 70)),
  levels = 51
)

#+ 
knn_grid

#+ 
# Tune kNN hyperparameters
# doParallel::registerDoParallel()
# set.seed(1234)
# knn_tune_rs <- tune_grid(
#   credit_wf %>% add_model(knn_spec),
#   resamples = credit_folds,
#   metrics = model_mets,
#   grid = knn_grid
# )
# 
# saveRDS(knn_tune_rs, file = here("out", "knn_tune_rs.rds"))
knn_tune_rs <- readRDS(here("out", "knn_tune_rs.rds"))

#+ 
# Examine AUROC for hyperparameters
knn_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  dplyr::select(mean, neighbors) %>%
  ggplot(aes(x = neighbors, y = mean)) +
  geom_point() +
  labs(
    y = "AUROC",
    title = "kNN - AUROC vs hyperparameter tuning values",
    subtitle = "Regular grid tuning"
  )

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "knn-auroc-tune.png")

#+ 
# Examine AUPRC for hyperparameters
knn_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "pr_auc") %>%
  dplyr::select(mean, neighbors) %>%
  ggplot(aes(x = neighbors, y = mean)) +
  geom_point() +
  labs(
    y = "AUPRC",
    title = "kNN - AUPRC vs hyperparameter tuning values",
    subtitle = "Regular grid tuning"
  )

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "knn-auprc-tune.png")

#+ 
# Save best hyperparameters based on AUROC and AUPRC
best_knn_auroc <- select_best(knn_tune_rs, metric = "roc_auc")
best_knn_auprc <- select_best(knn_tune_rs, metric = "pr_auc")
#' A high k isn't an issue as it is more biased towards underfitting (i.e. 
#' higher bias, but much lower variance) so AUC improves

#+ 
# Specify optimized kNN model - AUROC optimized
knn_final_auroc_spec <- finalize_model(
  knn_spec,
  best_knn_auroc
)

#+ 
# Specify optimized kNN model - AUPRC optimized
knn_final_auprc_spec <- finalize_model(
  knn_spec,
  best_knn_auprc
)

#+ 
# Fit kNN model to all folds in training data (resampling), saving certain metrics - AUROC optimized
# knn_final_auroc_rs <- credit_wf %>%
#   add_model(knn_final_auroc_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(knn_final_auroc_rs, file = here("out", "knn_final_auroc_rs.rds"))
knn_final_auroc_rs <- readRDS(here("out", "knn_final_auroc_rs.rds"))

#+ 
# Fit kNN model to all folds in training data (resampling), saving certain metrics - AUPRC optimized
# knn_final_auprc_rs <- credit_wf %>%
#   add_model(knn_final_auprc_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = model_mets,
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(knn_final_auprc_rs, file = here("out", "knn_final_auprc_rs.rds"))
knn_final_auprc_rs <- readRDS(here("out", "knn_final_auprc_rs.rds"))

#+ 
# Create ROC curve - AUROC optimized
knn_final_auroc_roc <- knn_final_auroc_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "kNN - AUROC")

#+ 
# Create Precision-Recall curve (PPV-Sensitivity) - AUROC optimized
knn_final_auroc_prc <- knn_final_auroc_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "kNN - AUROC")

#+ 
# Create tibble of metrics - AUROC optimized
knn_final_auroc_met <- knn_final_auroc_rs %>%
  collect_metrics() %>%
  mutate(model = "kNN - AUROC")

#+ 
# Create ROC curve - AUPRC optimized
knn_final_auprc_roc <- knn_final_auprc_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "kNN - AUPRC")

#+ 
# Create Precision-Recall curve (PPV-Sensitivity) - AUPRC optimized
knn_final_auprc_prc <- knn_final_auprc_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "kNN - AUPRC")

#+ 
# Create tibble of metrics - AUPRC optimized
knn_final_auprc_met <- knn_final_auprc_rs %>%
  collect_metrics() %>%
  mutate(model = "kNN - AUPRC")

#' # Model Evaluation
#' ## Metric Summaries
# Evaluate Metrics ---------------------------------------------------------

#+ 
all_auroc_met <- bind_rows(
  glm_met, glmnet_final_auroc_met, lda_met, qda_met,
  rf_final_auroc_met, xgb_final_auroc_met, bag_final_auroc_met,
  svmr_final_auroc_met, svmp_final_auroc_met, knn_final_auroc_met
)

#+ 
all_auprc_met <- bind_rows(
  glm_met, glmnet_final_auprc_met, lda_met, qda_met,
  rf_final_auprc_met, xgb_final_auprc_met, bag_final_auprc_met,
  svmr_final_auprc_met, svmp_final_auprc_met, knn_final_auprc_met
)

#+ 
# Rank all models by AUROC - AUROC optimized
all_auroc_met %>%
  filter(.metric == "roc_auc") %>%
  arrange(desc(mean))

#+ 
# Rank all models by AUROC - AUPRC optimized
all_auprc_met %>%
  filter(.metric == "roc_auc") %>%
  arrange(desc(mean))

#+ 
# Rank all models by sensitivity - AUROC optimized
all_auroc_met %>%
  filter(.metric == "sens") %>%
  arrange(desc(mean))

#+ 
# Rank all models by sensitivity - AUPRC optimized
all_auprc_met %>%
  filter(.metric == "sens") %>%
  arrange(desc(mean))

#+ 
# Rank all models by specificity - AUROC optimized
all_auroc_met %>%
  filter(.metric == "spec") %>%
  arrange(desc(mean))

#+ 
# Rank all models by specificity - AUPRC optimized
all_auprc_met %>%
  filter(.metric == "spec") %>%
  arrange(desc(mean))

#+ 
# Rank all models by accuracy - AUROC optimized
all_auroc_met %>%
  filter(.metric == "accuracy") %>%
  arrange(desc(mean))

#+ 
# Rank all models by accuracy - AUPRC optimized
all_auprc_met %>%
  filter(.metric == "accuracy") %>%
  arrange(desc(mean))

#' Important to note that the no information rate (the baseline accuracy because
#' it is achieved by always predicting the majority class "No fraud") is 99.82% 
#' (227443 / 227846). The highest accuracy achieved is by SVM-P optimizing for
#' AUPRC, and equal to `r all_auprc_met %>% filter(.metric == "accuracy") %>% arrange(desc(mean)) %>% pluck("mean", 1) * 100`%

#+ 
# Rank all models by AUPRC - AUROC optimized
all_auroc_met %>%
  filter(.metric == "pr_auc") %>%
  arrange(desc(mean))

#+ 
# Rank all models by AUPRC - AUPRC optimized
all_auprc_met %>%
  filter(.metric == "pr_auc") %>%
  arrange(desc(mean))

#+ 
# Rank all models by PPV - AUROC optimized
all_auroc_met %>%
  filter(.metric == "ppv") %>%
  arrange(desc(mean))

#+ 
# Rank all models by PPV - AUPRC optimized
all_auprc_met %>%
  filter(.metric == "ppv") %>%
  arrange(desc(mean))

#+ 
# Rank all models by NPV - AUROC optimized
all_auroc_met %>%
  filter(.metric == "npv") %>%
  arrange(desc(mean))

#+ 
# Rank all models by NPV - AUPRC optimized
all_auprc_met %>%
  filter(.metric == "npv") %>%
  arrange(desc(mean))

#' ## ROC Plots
# ROC Plots ---------------------------------------------------------------

#+
# Create a list to order the ROC labels based on AUROC
all_auroc_met %>%
  filter(.metric == "roc_auc") %>%
  arrange(desc(mean)) %>%
  select(model)

roc_auroc_list <- c(
  "SVM-R - AUROC",
  "XGBoost - AUROC",
  "Random Forest - AUROC",
  "GLMNET - AUROC",
  "SVM-P - AUROC",
  "Logistic Regression",
  "Bagged Tree - AUROC",
  "kNN - AUROC",
  "LDA",
  "QDA"
)

#+ dpi=300
# Plot ROC curves - AUROC optimized
bind_rows(
  glm_roc, glmnet_final_auroc_roc, lda_roc, qda_roc,
  rf_final_auroc_roc, xgb_final_auroc_roc, bag_final_auroc_roc,
  svmr_final_auroc_roc, svmp_final_auroc_roc, knn_final_auroc_roc
) %>%
  mutate(model = factor(
    model, 
    levels = roc_auroc_list)) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 2, col = "grey80") +
  coord_equal() +
  labs(
    title = "ROC plots for all models ",
    subtitle = "AUROC Optimized"
    )

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "roc-plot-auroc-all.png")

#+
# Create a list to order the ROC labels based on AUPRC
all_auprc_met %>%
  filter(.metric == "roc_auc") %>%
  arrange(desc(mean)) %>%
  select(model)

roc_auprc_list <- c(
  "Random Forest - AUPRC",
  "GLMNET - AUPRC",
  "Logistic Regression",
  "Bagged Tree - AUPRC",
  "kNN - AUPRC",
  "LDA",
  "QDA",
  "XGBoost - AUPRC",
  "SVM-P - AUPRC",
  "SVM-R - AUPRC"
)

#+ dpi=300
# Plot ROC curves - AUPRC optimized
bind_rows(
  glm_roc, glmnet_final_auprc_roc, lda_roc, qda_roc,
  rf_final_auprc_roc, xgb_final_auprc_roc, bag_final_auprc_roc,
  svmr_final_auprc_roc, svmp_final_auprc_roc, knn_final_auprc_roc
) %>%
  mutate(model = factor(
    model, 
    levels = roc_auprc_list)) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 2, col = "grey80") +
  coord_equal() +
  labs(
    title = "ROC plots for all models",
    subtitle = "AUPRC Optimized"
    )

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "roc-plot-auprc-all.png")

#' ## Precision Recall Curves
#+
# Precision Recall Curves -------------------------------------------------

#+
# Create a list to order the PRC labels based on AUPRC - AUROC optimized
all_auroc_met %>%
  filter(.metric == "pr_auc") %>%
  arrange(desc(mean)) %>%
  select(model)

prc_auroc_list <- c(
  "Random Forest - AUROC",
  "XGBoost - AUROC",
  "kNN - AUROC",
  "GLMNET - AUROC",
  "SVM-P - AUROC",
  "SVM-R - AUROC",
  "Logistic Regression",
  "QDA",
  "Bagged Tree - AUROC",
  "LDA"
)

#+ dpi=300
# Plot Precision-Recall curves - AUROC optimized
bind_rows(
  glm_prc, glmnet_final_auroc_prc, lda_prc, qda_prc,
  rf_final_auroc_prc, xgb_final_auroc_prc, bag_final_auroc_prc,
  svmr_final_auroc_prc, svmp_final_auroc_prc, knn_final_auroc_prc
) %>%
  mutate(model = factor(model, levels = prc_auroc_list)) %>%
  ggplot(aes(x = recall, y = precision, col = model)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 2, col = "grey80") +
  coord_equal() +
  labs(
    x = "Recall (Sensitivity)",
    y = "Precision (Positive Predictive Value)",
    title = "Precision (PPV) - Recall (Sens) curves for all models",
    subtitle = "AUROC Optimized"
  )

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "pr-plot-auroc-all.png")

#+
# Create a list to order the PRC labels based on AUPRC - AUPRC optimized
all_auprc_met %>%
  filter(.metric == "pr_auc") %>%
  arrange(desc(mean)) %>%
  select(model)

prc_auprc_list <- c(
  "Random Forest - AUPRC",
  "XGBoost - AUPRC",
  "SVM-P - AUPRC",
  "SVM-R - AUPRC",
  "kNN - AUPRC",
  "GLMNET - AUPRC",
  "Logistic Regression",
  "QDA",
  "Bagged Tree - AUPRC",
  "LDA"
)

#+ dpi=300
# Plot Precision-Recall curves - AUPRC optimized
bind_rows(
  glm_prc, glmnet_final_auprc_prc, lda_prc, qda_prc,
  rf_final_auprc_prc, xgb_final_auprc_prc, bag_final_auprc_prc,
  svmr_final_auprc_prc, svmp_final_auprc_prc, knn_final_auprc_prc
) %>%
  mutate(model = factor(model, levels = prc_auprc_list)) %>%
  ggplot(aes(x = recall, y = precision, col = model)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 2, col = "grey80") +
  coord_equal() +
  labs(
    x = "Recall (Sensitivity)",
    y = "Precision (Positive Predictive Value)",
    title = "Precision (PPV) - Recall (Sens) curves for all models",
    subtitle = "AUPRC Optimized"
  )

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "pr-plot-auprc-all.png")

#' ## Posterior Probability Distributions
#' ### Best Performers by Metrics

#+ dpi=300
# Compare predicted positive vs outcome - AUROC optimized
bind_rows(
  collect_predictions(svmr_final_auroc_rs) %>% mutate(model = "SVM-R"),
  collect_predictions(glm_rs) %>% mutate(model = "Logistic Regression"),
  collect_predictions(svmp_final_auroc_rs) %>% mutate(model = "SVM-P"),
  collect_predictions(rf_final_auroc_rs) %>% mutate(model = "Random Forest")
) %>%
  ggplot(aes(x = .pred_Fraud, fill = Class)) +
  geom_histogram(binwidth = 0.01) +
  scale_fill_ipsum() +
  labs(
    title = "Predicted probability of fraud distributions by known class",
    subtitle = "AUROC Optimized",
    caption = "Logistic Regression best sensitivity (0.924)
    Random Forest best AUPRC (0.783)
    SVM-P best specificity (0.999)
    SVM-P best accuracy (0.998) 
    SVM-P best PPV (0.525)
    SVM-R best AUROC (0.983)"
  ) +
  facet_wrap(~ Class + model, scales = "free_y", ncol = 4)

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "pred-dist-auroc-plot.png")

#+ dpi=300
# Compare predicted positive vs outcome - AUPRC optimized
bind_rows(
  collect_predictions(svmr_final_auprc_rs) %>% mutate(model = "SVM-R"),
  collect_predictions(glm_rs) %>% mutate(model = "Logistic Regression"),
  collect_predictions(svmp_final_auprc_rs) %>% mutate(model = "SVM-P"),
  collect_predictions(rf_final_auprc_rs) %>% mutate(model = "Random Forest")
) %>%
  ggplot(aes(x = .pred_Fraud, fill = Class)) +
  geom_histogram(binwidth = 0.01) +
  scale_fill_ipsum() +
  labs(
    title = "Predicted probability of fraud distributions by known class",
    subtitle = "AUPRC Optimized",
    caption = "Logistic Regression best sensitivity (0.924)
    Random Forest best AUROC (0.979)
    Random Forest best AUPRC (0.784)
    SVM-P best accuracy (0.999)
    SVM-P best PPV (0.868)
    SVM-R best specificity (1.00)"
  ) +
  facet_wrap(~ Class + model, scales = "free_y", ncol = 4)

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "pred-dist-auprc-plot.png")

#' ### Other models

#+ dpi=300
# Compare predicted positive vs outcome - AUROC optimized
bind_rows(
  collect_predictions(knn_final_auroc_rs) %>% mutate(model = "kNN"),
  collect_predictions(lda_rs) %>% mutate(model = "LDA"),
  collect_predictions(glmnet_final_auroc_rs) %>% mutate(model = "GLMNET"),
  collect_predictions(qda_rs) %>% mutate(model = "QDA")
) %>%
  ggplot(aes(x = .pred_Fraud, fill = Class)) +
  geom_histogram(binwidth = 0.01) +
  scale_fill_ipsum() +
  labs(
    title = "Predicted probability of fraud distributions by known class",
    subtitle = "AUROC Optimized"
  ) +
  facet_wrap(~ Class + model, scales = "free_y", ncol = 4)


#+ dpi=300
# Compare predicted positive vs outcome - AUPRC optimized
bind_rows(
  collect_predictions(knn_final_auprc_rs) %>% mutate(model = "kNN"),
  collect_predictions(lda_rs) %>% mutate(model = "LDA"),
  collect_predictions(glmnet_final_auprc_rs) %>% mutate(model = "GLMNET"),
  collect_predictions(qda_rs) %>% mutate(model = "QDA")
) %>%
  ggplot(aes(x = .pred_Fraud, fill = Class)) +
  geom_histogram(binwidth = 0.01) +
  scale_fill_ipsum() +
  labs(
    title = "Predicted probability of fraud distributions by known class",
    subtitle = "AUPRC Optimized"
  ) +
  facet_wrap(~ Class + model, scales = "free_y", ncol = 4)


#' ## Calibration Plots
# Calibration Plots -------------------------------------------------------

#' Calibration plots indicate how much the observed probabilities of an outcome
#' (Fraud) predicted in bins match the probabilities observed, i.e. the 0-0.1
#' probability bin would expect to see Fraud observed 5% of the time (the midpoint
#' of the bin, therefore average probability of the bin)

#+ 
# All probs tibble - AUROC optimized
train_auroc_preds <- glm_rs %>%
  collect_predictions() %>%
  dplyr::select(Class, .pred_Fraud) %>%
  transmute(
    Class = Class,
    glm = .pred_Fraud
  )

#+ 
train_auroc_preds$lda <- collect_predictions(lda_rs)$.pred_Fraud
train_auroc_preds$qda <- collect_predictions(qda_rs)$.pred_Fraud
train_auroc_preds$rf <- collect_predictions(rf_final_auroc_rs)$.pred_Fraud
train_auroc_preds$xgb <- collect_predictions(xgb_final_auroc_rs)$.pred_Fraud
train_auroc_preds$bag <- collect_predictions(bag_final_auroc_rs)$.pred_Fraud
train_auroc_preds$glmnet <- collect_predictions(glmnet_final_auroc_rs)$.pred_Fraud
train_auroc_preds$svmr <- collect_predictions(svmr_final_auroc_rs)$.pred_Fraud
train_auroc_preds$svmp <- collect_predictions(svmp_final_auroc_rs)$.pred_Fraud
train_auroc_preds$knn <- collect_predictions(knn_final_auroc_rs)$.pred_Fraud

#+ 
calib_auroc_df <- caret::calibration(
  Class ~ glm + lda + qda + rf + xgb + bag + glmnet + svmr + svmp + knn,
  data = train_auroc_preds,
  cuts = 10
)$data

#+ dpi=300
ggplot(calib_auroc_df, aes(
  x = midpoint,
  y = Percent,
  color = fct_reorder2(calibModelVar, midpoint, Percent)
)) +
  geom_abline(color = "grey30", linetype = 2) +
  geom_point(size = 1.5, alpha = 0.6) +
  geom_line(size = 1, alpha = 0.6) +
  labs(
    title = "Calibration plots for all models",
    subtitle = "AUROC Optimized",
    caption = "Perfect calibration lies on the diagonal",
    color = "Model"
  )

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "calib-auroc-plot-all.png")

#+ 
# All probs tibble - AUPRC optimized
train_auprc_preds <- glm_rs %>%
  collect_predictions() %>%
  dplyr::select(Class, .pred_Fraud) %>%
  transmute(
    Class = Class,
    glm = .pred_Fraud
  )

#+ 
train_auprc_preds$lda <- collect_predictions(lda_rs)$.pred_Fraud
train_auprc_preds$qda <- collect_predictions(qda_rs)$.pred_Fraud
train_auprc_preds$rf <- collect_predictions(rf_final_auprc_rs)$.pred_Fraud
train_auprc_preds$xgb <- collect_predictions(xgb_final_auprc_rs)$.pred_Fraud
train_auprc_preds$bag <- collect_predictions(bag_final_auprc_rs)$.pred_Fraud
train_auprc_preds$glmnet <- collect_predictions(glmnet_final_auprc_rs)$.pred_Fraud
train_auprc_preds$svmr <- collect_predictions(svmr_final_auprc_rs)$.pred_Fraud
train_auprc_preds$svmp <- collect_predictions(svmp_final_auprc_rs)$.pred_Fraud
train_auprc_preds$knn <- collect_predictions(knn_final_auprc_rs)$.pred_Fraud

#+ 
calib_auprc_df <- caret::calibration(
  Class ~ glm + lda + qda + rf + xgb + bag + glmnet + svmr + svmp + knn,
  data = train_auprc_preds,
  cuts = 10
)$data

#+ dpi=300
ggplot(calib_auprc_df, aes(
  x = midpoint,
  y = Percent,
  color = fct_reorder2(calibModelVar, midpoint, Percent)
)) +
  geom_abline(color = "grey30", linetype = 2) +
  geom_point(size = 1.5, alpha = 0.6) +
  geom_line(size = 1, alpha = 0.6) +
  labs(
    title = "Calibration plots for all models",
    subtitle = "AUPRC Optimized",
    caption = "Perfect calibration lies on the diagonal",
    color = "Model"
  )

#+ 
ggsave(plot = last_plot(), path = here("out"), filename = "calib-auprc-plot-all.png")

# Calibrating Models ------------------------------------------------------

#' Calibrating with monotonic function e.g. Platt scaling or isotonic regression
#' does not affect AUROC as ROC is based purely on ranking
#' (https://www.fharrell.com/post/mlconfusion/). Unlikely that accuracy will
#' be affected by either (https://www.youtube.com/watch?v=w3OPq0V8fr8)


#' ## Brier Scores
# Brier Scores ------------------------------------------------------------

#' As seen, there is a desire to evaluate models with severe class imbalances
#' using metrics other than accuracy based metrics. Frank Harrell suggests 
#' using proper scoring rules, such as the Brier score 
#' (https://www.fharrell.com/post/class-damage/)
#' (https://stats.stackexchange.com/questions/312780/why-is-accuracy-not-the-best-measure-for-assessing-classification-models)
#' (https://en.wikipedia.org/wiki/Scoring_rule)

#' Combination of calibration and accuracy.
#' 0 is perfect correct, 1 is perfectly wrong

#+ 
# Function
brier <- function(rs){
  preds <- collect_predictions(rs)
  # Reorder Class factor so can use .pred_Fraud as predicted outcome
  # Means there isn't a chance differnt models used different orders
  preds$Class <- fct_relevel(preds$Class, "None", "Fraud")
  
  f_t <- preds$.pred_Fraud
  o_t <- as.numeric(preds$Class) - 1
  print(paste0("Brier score for: ", deparse(substitute(rs))))
  
  mean((f_t - o_t)^2)
}

#+ 
# Logistic Regression
brier(glm_rs)

#+ 
# LDA
brier(lda_rs)

#+ 
# QDA
brier(qda_rs)

#+ 
# Random Forest - AUROC optimized
brier(rf_final_auroc_rs)

#+ 
# Random Forest - AUPRC optimized
brier(rf_final_auprc_rs)

#+ 
# XGBoost - AUROC optimized
brier(xgb_final_auroc_rs)

#+ 
# XGBoost - AUPRC optimized
brier(xgb_final_auprc_rs)

#+ 
# Bagged Trees - AUROC optimized
brier(bag_final_auroc_rs)

#+ 
# Bagged Trees - AUPRC optimized
brier(bag_final_auprc_rs)

#+ 
# GLMNET - AUROC optimized
brier(glmnet_final_auroc_rs)

#+ 
# GLMNET - AUPRC optimized
brier(glmnet_final_auprc_rs)

#+ 
# SVM-Radial - AUROC optimized
brier(svmr_final_auroc_rs)

#+ 
# SVM-Radial - AUPRC optimized
brier(svmr_final_auprc_rs)

#+ 
# SVM-Polynomial - AUROC optimized
brier(svmp_final_auroc_rs)

#+ 
# SVM-Polynomial - AUPRC optimized
brier(svmp_final_auprc_rs)

#+
# kNN - AUROC optimized
brier(knn_final_auroc_rs)

#+
# kNN - AUPRC optimized
brier(knn_final_auprc_rs)


#' # Test Data
# Test Data ---------------------------------------------------------------

#' RF had the highest AUPRC, indicating performance in identifying fraud cases,
#' so we will use it as the final model for the test data.

#+
# Evaluate the ROC for all folds in the training data
rf_final_auprc_rs %>%
  collect_predictions() %>%
  group_by(id) %>%
  roc_curve(Class, .pred_Fraud) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = id)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  labs(
    title = "ROCs for Final Model by Fold in Training Data",
    subtitle = "Random Forest Optimized using AUPRC",
    color = "Fold"
  ) +
  scale_color_ipsum()

#+
# Evaluate the PRC for all folds in the training data
rf_final_auprc_rs %>%
  collect_predictions() %>%
  group_by(id) %>%
  pr_curve(Class, .pred_Fraud) %>%
  ggplot(aes(x = recall, y = precision, color = id)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  labs(
    title = "PRCs for Final Model by Fold in Training Data",
    subtitle = "Random Forest Optimized using AUPRC",
    color = "Fold"
  ) +
  scale_color_ipsum()

#+
# Specify final model
credit_final_spec <- rf_final_auprc_spec

#+
# Fit final model to all training data and evaluate on test set
credit_final_rs <- credit_wf %>%
  add_model(credit_final_spec) %>%
  last_fit(credit_split, metrics = model_mets)

#+
collect_metrics(credit_final_rs)

#+
collect_predictions(credit_final_rs) %>%
  conf_mat(Class, .pred_class)

#+
# Compare ROCs for training vs testing in final RF model
credit_final_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Test Data") %>%
  bind_rows(rf_final_auprc_roc) %>%
  mutate(model = case_when(
    model == "Random Forest - AUPRC" ~ "Training Data", 
    TRUE ~ model
    )) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  labs(
    title = "ROCs for Final Model in Training and Test Data",
    subtitle = "Random Forest Optimized for AUPRC",
    color = "Data Type"
  ) +
  scale_color_ipsum()

#+
# Compare PRCs for training vs testing in final RF model
credit_final_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Test Data") %>%
  bind_rows(rf_final_auprc_prc) %>%
  mutate(model = case_when(
    model == "Random Forest - AUPRC" ~ "Training Data", 
    TRUE ~ model
  )) %>%
  ggplot(aes(x = recall, y = precision, color = model)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  labs(
    title = "PRC for Final Model in Training and Test Data",
    subtitle = "Random Forest Optimized for AUPRC",
    x = "Recall (Sensitivity)",
    y = "Precision (Positive Predictive Value)",
    color = "Data Type"
  ) +
  scale_color_ipsum()
