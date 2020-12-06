#' ---
#' title: "STAT 508 Final Project Analysis"
#' author: "Callum Arnold"
#' output:
#'   pdf_document
#' ---

# Set Up ------------------------------------------------------------------

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

RNGkind(sample.kind = "Rounding")
set.seed(1)

theme_set(theme_ipsum())

credit <- as_tibble(read_csv(here("data", "creditcard.csv")))

credit <- credit %>%
  mutate(
    log_amount = log(Amount + 1),
    Class = case_when(
      Class == 0 ~ "None",
      Class == 1 ~ "Fraud"
    )
  ) %>%
  mutate(across(.cols = Class, .fns = factor)) %>%
  select(-Amount)

tabyl(credit$Class)

levels(credit$Class)

# Pre-Processing ----------------------------------------------------------


#' Because there's a severe class imbalance, we should use subsampling (either oversampling or undersampling).

#' Subsampling has a few important points regarding the [workflow](https://www.tidymodels.org/learn/models/sub-sampling/):

#' - It is extremely important that subsampling occurs inside of resampling. Otherwise, the resampling process can produce poor estimates of model performance.
#' - The subsampling process should only be applied to the analysis set. The assessment set should reflect the event rates seen “in the wild” and, for this reason, the skip argument to step_downsample() and other subsampling recipes steps has a default of TRUE.

set.seed(1234)
credit_split <- initial_split(credit, strata = "Class", prop = 0.8)

credit_train <- training(credit_split)
credit_test <- testing(credit_split)

tabyl(credit_train$Class)
tabyl(credit_test$Class)

# Create fold so same folds analyzed for every method
set.seed(1234)
credit_folds <- vfold_cv(credit_train, v = 10)

# Create recipe so all folds are undergo same pre-processing
credit_rec <-
  recipe(Class ~ ., data = credit_train) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_zv(all_numeric()) %>%
  step_normalize(Time, log_amount) %>%
  step_downsample(Class)

# Create workflow so all models can replicate same analysis methods
credit_wf <- workflow() %>%
  add_recipe(credit_rec)

# Create list of metrics to record in model fits
model_mets <- metric_set(
  roc_auc, accuracy, sensitivity, specificity, j_index,
  ppv, npv, pr_auc
)

# Logistic Regression -----------------------------------------------------

# Specify logistic model
glm_spec <- logistic_reg() %>%
  set_engine("glm")

# Fit logistic model to all folds in training data (resampling), saving certain metrics
doParallel::registerDoParallel()
glm_rs <- credit_wf %>%
  add_model(glm_spec) %>%
  fit_resamples(
    resamples = credit_folds,
    metrics = model_mets,
    control = control_resamples(save_pred = TRUE)
  )

saveRDS(glm_rs, here("out", "glm_rs.rds"))
glm_rs <- readRDS(here("out", "glm_rs.rds"))

# Examine which variables are most important
glm_spec %>%
  set_engine("glm") %>%
  fit(Class ~ .,
    data = juice(prep(credit_rec))
  ) %>%
  vip(geom = "point") +
  labs(title = "Logistic Regression VIP")

ggsave(plot = last_plot(), path = here("out"), filename = "glm-vip.png")

# Create roc curve
glm_roc <- glm_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Logistic Regression")

# Create Precision-Recall curve (PPV-Sensitivity)
glm_prc <- glm_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Logistic Regression")

# Create tibble of metrics
glm_met <- glm_rs %>%
  collect_metrics() %>%
  mutate(model = "Logistic Regression")

# LDA ---------------------------------------------------------------------

# Specify LDA model
lda_spec <- discrim_linear() %>%
  set_engine("MASS")

# Fit logistic model to all folds in training data (resampling), saving certain metrics
doParallel::registerDoParallel()
lda_rs <- credit_wf %>%
  add_model(lda_spec) %>%
  fit_resamples(
    resamples = credit_folds,
    metrics = model_mets,
    control = control_resamples(save_pred = TRUE)
  )
#
saveRDS(lda_rs, here("out", "lda_rs.rds"))
lda_rs <- readRDS(here("out", "lda_rs.rds"))

# Create roc curve
lda_roc <- lda_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "LDA")

# Create Precision-Recall curve (PPV-Sensitivity)
lda_prc <- lda_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "LDA")

# Create tibble of metrics
lda_met <- lda_rs %>%
  collect_metrics() %>%
  mutate(model = "LDA")


# QDA ---------------------------------------------------------------------

qda_spec <- discrim_regularized(frac_common_cov = 0, frac_identity = 0) %>%
  set_engine("klaR")


# Fit QDA model to all folds in training data (resampling), saving certain metrics
doParallel::registerDoParallel()
qda_rs <- credit_wf %>%
  add_model(qda_spec) %>%
  fit_resamples(
    resamples = credit_folds,
    metrics = model_mets,
    control = control_resamples(save_pred = TRUE)
  )

saveRDS(qda_rs, here("out", "qda_rs.rds"))
qda_rs <- readRDS(here("out", "qda_rs.rds"))

# Create roc curve
qda_roc <- qda_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "QDA")

# Create Precision-Recall curve (PPV-Sensitivity)
qda_prc <- qda_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "QDA")

# Create tibble of metrics
qda_met <- qda_rs %>%
  collect_metrics() %>%
  mutate(model = "QDA")


# Random Forest -----------------------------------------------------------

# Specify random forest model
rf_spec <- rand_forest(
  mtry = tune(),
  trees = 500,
  min_n = tune()
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Tune random forest hyperparameters
doParallel::registerDoParallel()
set.seed(1234)
rf_tune_rs <- tune_grid(
  credit_wf %>% add_model(rf_spec),
  resamples = credit_folds,
  metrics = model_mets,
  grid = 20
)

saveRDS(rf_tune_rs, file = here("out", "rf_tune_rs.rds"))
rf_tune_rs <- readRDS(here("out", "rf_tune_rs.rds"))

rf_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, min_n, mtry) %>%
  pivot_longer(
    cols = min_n:mtry,
    values_to = "value",
    names_to = "parameter"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(
    x = NULL, y = "AUC",
    title = "Random Forest - AUROC vs hyperparameter tuning values",
    subtitle = "Initial tuning"
  )

ggsave(plot = last_plot(), path = here("out"), filename = "rf-initial-roc-tune.png")

#' We can see that lower values of `min_n` are better, and no pattern with `mtry`.
#' Let's create a regular grid to do a finer optimization.

rf_grid <- grid_regular(
  mtry(range = c(0, 25)),
  min_n(range = c(1, 10)),
  levels = 6
)

rf_grid

# Fit Random Forest with regular tuning grid that is more focussed
doParallel::registerDoParallel()
set.seed(1234)
rf_reg_tune_rs <- tune_grid(
  credit_wf %>% add_model(rf_spec),
  resamples = credit_folds,
  metrics = model_mets,
  grid = rf_grid
)

saveRDS(rf_reg_tune_rs, file = here("out", "rf_reg_tune_rs.rds"))
rf_reg_tune_rs <- readRDS(here("out", "rf_reg_tune_rs.rds"))

# Examine AUC for hyperparameters
rf_reg_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(x = mtry, y = mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(
    y = "AUC",
    title = "Random Forest - AUROC vs hyperparameter tuning values",
    subtitle = "Regular grid tuning"
  )

ggsave(plot = last_plot(), path = here("out"), filename = "rf-grid-roc-tune.png")

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

ggsave(plot = last_plot(), path = here("out"), filename = "rf-grid-acc-tune.png")

#' We can see from the plot of AUC that the best combination is `min_n = 1`, and
#' `mtry = 10`. There seems to be a decline in accuracy from `mtry = 5`, however,
#' this is likely due to reduced sensitivity and improved specificity, which is
#' the opposite of what we're interested in given the class imbalance.
#' It is generally accepted that good starting points are `mtry = sqrt(p)` (c. 5)
#' and `min_n = 1` for classification models (https://bradleyboehmke.github.io/HOML/random-forest.html)

best_rf_auc <- select_best(rf_reg_tune_rs, "roc_auc")

rf_final_spec <- finalize_model(
  rf_spec,
  best_rf_auc
)

# Examine which variables are most important
rf_final_spec %>%
  set_engine("ranger", importance = "permutation") %>%
  fit(Class ~ .,
    data = juice(prep(credit_rec))
  ) %>%
  vip(geom = "point") +
  labs(title = "Random Forest VIP")

ggsave(plot = last_plot(), path = here("out"), filename = "rf-final-vip.png")

#' Important to note that PCA is unsupervised so only looks at relevance to the
#' variance observed in the predictors, not at their relevance to the outcome,
#' so not necessary that PC1 would be the most important PC in predicting Class

# Fit random forest model to all folds in training data (resampling), saving certain metrics
rf_final_rs <- credit_wf %>%
  add_model(rf_final_spec) %>%
  fit_resamples(
    resamples = credit_folds,
    metrics = model_mets,
    control = control_resamples(save_pred = TRUE)
  )

saveRDS(rf_final_rs, file = here("out", "rf_final_rs.rds"))
rf_final_rs <- readRDS(here("out", "rf_final_rs.rds"))


# Create roc curve
rf_final_roc <- rf_final_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Random Forest")

# Create Precision-Recall curve (PPV-Sensitivity)
rf_final_prc <- rf_final_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Random Forest")

# Create tibble of metrics
rf_final_met <- rf_final_rs %>%
  collect_metrics() %>%
  mutate(model = "Random Forest")


# XGBoost -----------------------------------------------------------------

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

# Tune XGBoost hyperparameters using space filling parameter grid
doParallel::registerDoParallel()
set.seed(1234)
xgb_tune_rs <- tune_grid(
  credit_wf %>% add_model(xgb_spec),
  resamples = credit_folds,
  metrics = model_mets,
  grid = xgb_grid
)

saveRDS(xgb_tune_rs, file = here("out", "xgb_tune_rs.rds"))
xgb_tune_rs <- readRDS(here("out", "xgb_tune_rs.rds"))

# Examine AUC for hyperparameters
xgb_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, mtry:sample_size) %>%
  pivot_longer(
    mtry:sample_size,
    names_to = "parameter",
    values_to = "value"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point() +
  labs(
    y = "AUC",
    title = "XGBoost - AUROC vs hyperparameter tuning values",
    subtitle = "LHS grid tuning"
  ) +
  facet_wrap(~parameter, scales = "free_x")

ggsave(plot = last_plot(), path = here("out"), filename = "xgb-roc-tune.png")

show_best(xgb_tune_rs, "roc_auc")

best_xgb_auc <- select_best(xgb_tune_rs, "roc_auc")

xgb_final_spec <- finalize_model(
  xgb_spec,
  best_xgb_auc
)

# Examine which variables are most important
xgb_final_spec %>%
  set_engine("xgboost", importance = "permutation") %>%
  fit(Class ~ .,
    data = juice(prep(credit_rec))
  ) %>%
  vip(geom = "point") +
  labs(title = "XGBoost VIP")

ggsave(plot = last_plot(), path = here("out"), filename = "xgb-final-vip.png")

# Fit XGBoost model to all folds in training data (resampling), saving certain metrics
xgb_final_rs <- credit_wf %>%
  add_model(xgb_final_spec) %>%
  fit_resamples(
    resamples = credit_folds,
    metrics = model_mets,
    control = control_resamples(save_pred = TRUE)
  )

saveRDS(xgb_final_rs, file = here("out", "xgb_final_rs.rds"))
xgb_final_rs <- readRDS(here("out", "xgb_final_rs.rds"))

# Create roc curve
xgb_final_roc <- xgb_final_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "XGBoost")

# Create Precision-Recall curve (PPV-Sensitivity)
xgb_final_prc <- xgb_final_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "XGBoost")

# Create tibble of metrics
xgb_final_met <- xgb_final_rs %>%
  collect_metrics() %>%
  mutate(model = "XGBoost")

# Bagged Tree -------------------------------------------------------------

# Specify bagged tree model
bag_spec <- bag_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# Create space filling parameter latin hypercube grid - regular grid too slow
bag_grid <- grid_latin_hypercube(
  cost_complexity(),
  tree_depth(),
  min_n(),
  size = 20
)

# Tune bagged tree hyperparameters using space filling parameter grid
doParallel::registerDoParallel()
set.seed(1234)
bag_tune_rs <- tune_grid(
  credit_wf %>% add_model(bag_spec),
  resamples = credit_folds,
  metrics = model_mets,
  grid = bag_grid
)

saveRDS(bag_tune_rs, file = here("out", "bag_tune_rs.rds"))
bag_tune_rs <- readRDS(here("out", "bag_tune_rs.rds"))

# Examine AUC for hyperparameters
bag_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, cost_complexity:min_n) %>%
  pivot_longer(
    cost_complexity:min_n,
    names_to = "parameter",
    values_to = "value"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point() +
  labs(
    y = "AUC",
    title = "Bagged Tree - AUROC vs hyperparameter tuning values",
    subtitle = "LHS grid tuning"
  ) +
  facet_wrap(~parameter, scales = "free_x")

ggsave(plot = last_plot(), path = here("out"), filename = "bag-roc-tune.png")

show_best(bag_tune_rs, "roc_auc")

# Select best parameters from tuning grid based on AUC
best_bag_auc <- select_best(bag_tune_rs, "roc_auc")

# Specify optimized bagged tree model
bag_final_spec <- finalize_model(
  bag_spec,
  best_bag_auc
)

# Examine which variables are most important
bag_imp <- bag_final_spec %>%
  set_engine("rpart") %>%
  fit(Class ~ .,
    data = juice(prep(credit_rec))
  )

bag_imp$fit$imp %>%
  mutate(term = fct_reorder(term, value)) %>%
  ggplot(aes(x = value, y = term)) +
  geom_point() +
  labs(title = "Bagged Tree VIP")

ggsave(plot = last_plot(), path = here("out"), filename = "bag-final-vip.png")

# Fit bagged tree model to all folds in training data (resampling), saving certain metrics
bag_final_rs <- credit_wf %>%
  add_model(bag_final_spec) %>%
  fit_resamples(
    resamples = credit_folds,
    metrics = model_mets,
    control = control_resamples(save_pred = TRUE)
  )

saveRDS(bag_final_rs, file = here("out", "bag_final_rs.rds"))
bag_final_rs <- readRDS(here("out", "bag_final_rs.rds"))

# Create roc curve
bag_final_roc <- bag_final_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Bagged Tree")

# Create Precision-Recall curve (PPV-Sensitivity)
bag_final_prc <- bag_final_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "Bagged Tree")

# Create tibble of metrics
bag_final_met <- bag_final_rs %>%
  collect_metrics() %>%
  mutate(model = "Bagged Tree")

# GLMNET ------------------------------------------------------------------

# Specify GLMNET model
glmnet_spec <- logistic_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")

# Create grid to tune penalty and mixture (Lasso vs Ridge)
glmnet_grid <- grid_latin_hypercube(
  penalty(),
  mixture(),
  size = 20
)

# Tune GLMNET hyperparameters
doParallel::registerDoParallel()
set.seed(1234)
glmnet_tune_rs <- tune_grid(
  credit_wf %>% add_model(glmnet_spec),
  resamples = credit_folds,
  metrics = model_mets,
  grid = glmnet_grid
)

saveRDS(glmnet_tune_rs, file = here("out", "glmnet_tune_rs.rds"))
glmnet_tune_rs <- readRDS(here("out", "glmnet_tune_rs.rds"))

# Examine AUC for hyperparameters
glmnet_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, penalty, mixture) %>%
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

ggsave(plot = last_plot(), path = here("out"), filename = "glmnet-roc-tune.png")

best_glmnet_auc <- select_best(glmnet_tune_rs, metric = "roc_auc")

# Specify optimized GLMNET model
glmnet_final_spec <- finalize_model(
  glmnet_spec,
  best_glmnet_auc
)

# Examine which variables are most important
glmnet_final_spec %>%
  set_engine("glmnet", importance = "permutation") %>%
  fit(Class ~ .,
    data = juice(prep(credit_rec))
  ) %>%
  vip(geom = "point") +
  labs(title = "GLMNET VIP")

ggsave(plot = last_plot(), path = here("out"), filename = "glmnet-final-vip.png")

# Fit GLMNET model to all folds in training data (resampling), saving certain metrics
glmnet_final_rs <- credit_wf %>%
  add_model(glmnet_final_spec) %>%
  fit_resamples(
    resamples = credit_folds,
    metrics = model_mets,
    control = control_resamples(save_pred = TRUE)
  )

saveRDS(glmnet_final_rs, file = here("out", "glmnet_final_rs.rds"))
glmnet_final_rs <- readRDS(here("out", "glmnet_final_rs.rds"))

# Create roc curve
glmnet_final_roc <- glmnet_final_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "GLMNET")

# Create Precision-Recall curve (PPV-Sensitivity)
glmnet_final_prc <- glmnet_final_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "GLMNET")

# Create tibble of metrics
glmnet_final_met <- glmnet_final_rs %>%
  collect_metrics() %>%
  mutate(model = "GLMNET")

# SVM - Radial ------------------------------------------------------------

# Specify SVM-Radial model
svmr_spec <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svmr_grid <- grid_latin_hypercube(
  cost(),
  rbf_sigma(),
  size = 20
)

# Tune SVM-Radial hyperparameters
doParallel::registerDoParallel()
set.seed(1234)
svmr_tune_rs <- tune_grid(
  credit_wf %>% add_model(svmr_spec),
  resamples = credit_folds,
  metrics = model_mets,
  grid = svmr_grid
)

saveRDS(svmr_tune_rs, file = here("out", "svmr_tune_rs.rds"))
svmr_tune_rs <- readRDS(here("out", "svmr_tune_rs.rds"))

# Examine AUC for hyperparameters
svmr_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, cost, rbf_sigma) %>%
  pivot_longer(
    cost:rbf_sigma,
    names_to = "parameter",
    values_to = "value"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point() +
  labs(
    y = "AUC",
    title = "SVM Radial - AUROC vs hyperparameter tuning values",
    subtitle = "LHS grid tuning"
  ) +
  facet_wrap(~parameter, scales = "free_x")

ggsave(plot = last_plot(), path = here("out"), filename = "svmr-roc-tune.png")

best_svmr_auc <- select_best(svmr_tune_rs, metric = "roc_auc")

# Specify optimized svm model
svmr_final_spec <- finalize_model(
  svmr_spec,
  best_svmr_auc
)

# Fit svm model to all folds in training data (resampling), saving certain metrics
svmr_final_rs <- credit_wf %>%
  add_model(svmr_final_spec) %>%
  fit_resamples(
    resamples = credit_folds,
    metrics = model_mets,
    control = control_resamples(save_pred = TRUE)
  )

saveRDS(svmr_final_rs, file = here("out", "svmr_final_rs.rds"))
svmr_final_rs <- readRDS(here("out", "svmr_final_rs.rds"))

# Create roc curve
svmr_final_roc <- svmr_final_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "SVM-R")

# Create Precision-Recall curve (PPV-Sensitivity)
svmr_final_prc <- svmr_final_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "SVM-R")

# Create tibble of metrics
svmr_final_met <- svmr_final_rs %>%
  collect_metrics() %>%
  mutate(model = "SVM-R")


# SVM - Polynomial --------------------------------------------------------

# Specify SVM-Polynomial model
svmp_spec <- svm_poly(
  cost = tune(),
  degree = tune(),
  scale_factor = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svmp_grid <- grid_latin_hypercube(
  cost(),
  degree(),
  scale_factor(),
  size = 20
)

# Tune SVM-P hyperparameters
doParallel::registerDoParallel()
set.seed(1234)
svmp_tune_rs <- tune_grid(
  credit_wf %>% add_model(svmp_spec),
  resamples = credit_folds,
  metrics = model_mets,
  grid = svmp_grid
)

saveRDS(svmp_tune_rs, file = here("out", "svmp_tune_rs.rds"))
svmp_tune_rs <- readRDS(here("out", "svmp_tune_rs.rds"))

# Examine AUC for hyperparameters
svmp_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, cost:scale_factor) %>%
  pivot_longer(
    cost:scale_factor,
    names_to = "parameter",
    values_to = "value"
  ) %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point() +
  labs(
    y = "AUC",
    title = "SVM Polynomial - AUROC vs hyperparameter tuning values",
    subtitle = "LHS grid tuning"
  ) +
  facet_wrap(~parameter, scales = "free_x")

ggsave(plot = last_plot(), path = here("out"), filename = "svmp-roc-tune.png")

best_svmp_auc <- select_best(svmp_tune_rs, metric = "roc_auc")

# Specify optimized svm model
svmp_final_spec <- finalize_model(
  svmp_spec,
  best_svmp_auc
)

# Fit svm model to all folds in training data (resampling), saving certain metrics
svmp_final_rs <- credit_wf %>%
  add_model(svmp_final_spec) %>%
  fit_resamples(
    resamples = credit_folds,
    metrics = model_mets,
    control = control_resamples(save_pred = TRUE)
  )

saveRDS(svmp_final_rs, file = here("out", "svmp_final_rs.rds"))
svmp_final_rs <- readRDS(here("out", "svmp_final_rs.rds"))

# Create roc curve
svmp_final_roc <- svmp_final_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "SVM-P")

# Create Precision-Recall curve (PPV-Sensitivity)
svmp_final_prc <- svmp_final_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "SVM-P")

# Create tibble of metrics
svmp_final_met <- svmp_final_rs %>%
  collect_metrics() %>%
  mutate(model = "SVM-P")


# kNN ---------------------------------------------------------------------

# Specify kNN model
knn_spec <- nearest_neighbor(
  neighbors = tune()
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

knn_grid <- grid_regular(
  neighbors(range = c(1, 70)),
  levels = 51
)

knn_grid

# Tune kNN hyperparameters
doParallel::registerDoParallel()
set.seed(1234)
knn_tune_rs <- tune_grid(
  credit_wf %>% add_model(knn_spec),
  resamples = credit_folds,
  metrics = model_mets,
  grid = knn_grid
)

saveRDS(knn_tune_rs, file = here("out", "knn_tune_rs.rds"))
knn_tune_rs <- readRDS(here("out", "knn_tune_rs.rds"))

# Examine AUC for hyperparameters
knn_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, neighbors) %>%
  ggplot(aes(x = neighbors, y = mean)) +
  geom_point() +
  labs(
    y = "AUC",
    title = "kNN - AUROC vs hyperparameter tuning values",
    subtitle = "Regular grid tuning"
  )

ggsave(plot = last_plot(), path = here("out"), filename = "knn-roc-tune.png")

best_knn_auc <- select_best(knn_tune_rs, metric = "roc_auc")

# Specify optimized svm model
knn_final_spec <- finalize_model(
  knn_spec,
  best_knn_auc
)

# Fit kNN model to all folds in training data (resampling), saving certain metrics
knn_final_rs <- credit_wf %>%
  add_model(knn_final_spec) %>%
  fit_resamples(
    resamples = credit_folds,
    metrics = model_mets,
    control = control_resamples(save_pred = TRUE)
  )

saveRDS(knn_final_rs, file = here("out", "knn_final_rs.rds"))
knn_final_rs <- readRDS(here("out", "knn_final_rs.rds"))

# Create roc curve
knn_final_roc <- knn_final_rs %>%
  collect_predictions() %>%
  roc_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "kNN")

# Create Precision-Recall curve (PPV-Sensitivity)
knn_final_prc <- knn_final_rs %>%
  collect_predictions() %>%
  pr_curve(truth = Class, .pred_Fraud) %>%
  mutate(model = "kNN")

# Create tibble of metrics
knn_final_met <- knn_final_rs %>%
  collect_metrics() %>%
  mutate(model = "kNN")


# Evaluate Models ---------------------------------------------------------

glm_met
lda_met
rf_final_met
xgb_final_met
bag_final_met
glmnet_final_met
svmr_final_met
svmp_final_met
knn_final_met

all_met <- bind_rows(
  glm_met, lda_met, rf_final_met, xgb_final_met, bag_final_met,
  glmnet_final_met, svmr_final_met, svmp_final_met, knn_final_met
)

# Rank all models by AUC
all_met %>%
  filter(.metric == "roc_auc") %>%
  arrange(desc(mean))

# Rank all models by sensitivity
all_met %>%
  filter(.metric == "sens") %>%
  arrange(desc(mean))

# Rank all models by specificity
all_met %>%
  filter(.metric == "spec") %>%
  arrange(desc(mean))

# Rank all models by accuracy
all_met %>%
  filter(.metric == "accuracy") %>%
  arrange(desc(mean))

# Rank all models by AUPRC
all_met %>%
  filter(.metric == "pr_auc") %>%
  arrange(desc(mean))

# Rank all models by PPV
all_met %>%
  filter(.metric == "ppv") %>%
  arrange(desc(mean))

# Rank all models by NPV
all_met %>%
  filter(.metric == "npv") %>%
  arrange(desc(mean))

# Plot ROC curves
bind_rows(
  glm_roc, lda_roc, qda_roc, glmnet_final_roc,
  rf_final_roc, xgb_final_roc, bag_final_roc,
  svmr_final_roc, svmp_final_roc, knn_final_roc
) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 2, col = "grey80") +
  coord_equal() +
  labs(title = "ROC plots for all models")

ggsave(plot = last_plot(), path = here("out"), filename = "roc-plot-all.png")

# Plot Precision-Recall curves
bind_rows(
  glm_prc, lda_prc, qda_prc, glmnet_final_prc,
  rf_final_prc, xgb_final_prc, bag_final_prc,
  svmr_final_prc, svmp_final_prc, knn_final_prc
) %>%
  ggplot(aes(x = recall, y = precision, col = model)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 2, col = "grey80") +
  coord_equal() +
  labs(
    x = "Recall (Sensitivity)",
    y = "Precision (Positive Predictive Value)",
    title = "Precision (PPV) - Recall (Sens) curves for all models"
  )

ggsave(plot = last_plot(), path = here("out"), filename = "pr-plot-all.png")

# Compare predicted positive vs outcome
bind_rows(
  collect_predictions(svmr_final_rs) %>% mutate(model = "SVM-R"),
  collect_predictions(glm_rs) %>% mutate(model = "Logistic Regression"),
  collect_predictions(svmp_final_rs) %>% mutate(model = "SVM-P")
) %>%
  ggplot(aes(x = .pred_Fraud, fill = Class)) +
  geom_histogram(binwidth = 0.01) +
  scale_fill_ipsum() +
  labs(
    title = "Predicted probability of fraud distributions by known class",
    caption = "SVM-P best specificity (0.999) \n SVM-R best AUC (0.983) \n SVM-P best accuracy (0.998) \n Logistic Regression best sensitivity (0.924)"
  ) +
  facet_wrap(~ Class + model, scales = "free_y", ncol = 3)

ggsave(plot = last_plot(), path = here("out"), filename = "pred-dist-plot.png")


# Calibration Plots -------------------------------------------------------

#' Calibration plots indicate how much the observed probabilities of an outcome
#' (Fraud) predicted in bins match the probabilities observed, i.e. the 0-0.1
#' probability bin would expect to see Fraud observed 5% of the time (the midpoint
#' of the bin, therefore average probability of the bin)

# All probs tibble
train_preds <- glm_rs %>%
  collect_predictions() %>%
  select(Class, .pred_Fraud) %>%
  transmute(
    Class = Class,
    glm = .pred_Fraud
  )

train_preds$lda <- collect_predictions(lda_rs)$.pred_Fraud
train_preds$qda <- collect_predictions(qda_rs)$.pred_Fraud
train_preds$rf <- collect_predictions(rf_final_rs)$.pred_Fraud
train_preds$xgb <- collect_predictions(xgb_final_rs)$.pred_Fraud
train_preds$bag <- collect_predictions(bag_final_rs)$.pred_Fraud
train_preds$glmnet <- collect_predictions(glmnet_final_rs)$.pred_Fraud
train_preds$svmr <- collect_predictions(svmr_final_rs)$.pred_Fraud
train_preds$svmp <- collect_predictions(svmp_final_rs)$.pred_Fraud
train_preds$knn <- collect_predictions(knn_final_rs)$.pred_Fraud

calib_df <- caret::calibration(
  Class ~ glm + lda + qda + rf + xgb + bag + glmnet + svmr + svmp + knn,
  data = train_preds,
  cuts = 10
)$data

ggplot(calib_df, aes(x = midpoint, y = Percent, color = calibModelVar)) +
  geom_abline(color = "grey30", linetype = 2) +
  geom_point(size = 1.5, alpha = 0.6) +
  geom_line(size = 1, alpha = 0.6) +
  labs(
    title = "Calibration plots for all models",
    caption = "Perfect calibration lies on the diagonal"
  )

ggsave(plot = last_plot(), path = here("out"), filename = "calib-plot-all.png")

# Brier Scores ------------------------------------------------------------

#' Combination of calibration and accuracy.
#' 0 is perfect correct, 1 is perfectly wrong

# Logistic Regression
glm_f_t <- collect_predictions(glm_rs)$.pred_None
glm_o_t <- as.numeric(collect_predictions(glm_rs)$Class) - 1
mean((glm_f_t - glm_o_t)^2)

# Random Forest
rf_f_t <- collect_predictions(rf_final_rs)$.pred_None
rf_o_t <- as.numeric(collect_predictions(rf_final_rs)$Class) - 1
mean((rf_f_t - rf_o_t)^2)

# Logistic Regression
glm_preds <- glm_rs %>%
  collect_predictions()

glm_cal <- caret::calibration(Class ~ .pred_Fraud, data = glm_preds)

ggplot(glm_cal)

glm_preds %>% tabyl(Class, .pred_class)

# Random Forest
rf_final_cal <- caret::calibration(Class ~ .pred_Fraud, data = collect_predictions(rf_final_rs))

ggplot(rf_final_cal)

# SVM-P
svmp_final_cal <- caret::calibration(Class ~ .pred_Fraud, data = collect_predictions(svmp_final_rs))

ggplot(svmp_final_cal)

# SVM-R
svmr_final_cal <- caret::calibration(Class ~ .pred_Fraud, data = collect_predictions(svmr_final_rs))

ggplot(svmr_final_cal)

# kNN
ggplot(caret::calibration(Class ~ .pred_Fraud, data = collect_predictions(knn_final_rs)))

# #' GLM has higher sensitivity, so will be better at detecting fraud.
#
# glm_rs %>%
#   collect_predictions() %>%
#   group_by(id) %>%
#   roc_curve(Class, .pred_Fraud) %>%
#   autoplot()
#
# # Fit final model to all training data and evaluate on test set
# credit_final <- credit_wf %>%
#   add_model(glm_spec) %>%
#   last_fit(credit_split)
#
# collect_metrics(credit_final)
#
# collect_predictions(credit_final) %>%
#   conf_mat(Class, .pred_class)
#
# credit_final %>%
#   pull(.workflow) %>%
#   pluck(1) %>%
#   tidy(exponentiate = FALSE) %>%
#   arrange(estimate) %>%
#   kable(digits = 3)
#
# credit_final %>%
#   pull(.workflow) %>%
#   pluck(1) %>%
#   tidy(exponentiate = FALSE) %>%
#   filter(term != "(Intercept)") %>%
#   ggplot(aes(estimate, fct_reorder(term, estimate))) +
#   geom_point() +
#   geom_errorbar(aes(
#     xmin = estimate - std.error,
#     xmax = estimate + std.error,
#     width = 0.2,
#     alpha = 0.7
#   )) +
#   geom_vline(xintercept = 1, color = "grey50", lty = 2)


#' Seems like `V17` has a very large positive impact on being predicted Fraud.


# Test Data ---------------------------------------------------------------

credit_wf %>%
  add_model(glm_spec) %>%
  last_fit(credit_split, metrics = metric_set(roc_auc, sens, spec, ppv, npv, pr_auc, kap)) %>%
  collect_metrics()
