#' ---
#' title: "STAT 508 Final Project Analysis"
#' author: "Callum Arnold"
#' output:
#'   pdf_document
#' ---

# Set Up ------------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(themis)
library(kknn)
library(here)
library(kableExtra)
library(hrbrthemes)
library(janitor)

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

# Logistic Regression -----------------------------------------------------

# Specify logistic model
glm_spec <- logistic_reg() %>%
  set_engine("glm")

# Fit logistic model to all folds in training data (resampling), saving certain metrics
# doParallel::registerDoParallel()
# glm_rs <- credit_wf %>%
#   add_model(glm_spec) %>%
#   fit_resamples(
#     resamples = credit_folds,
#     metrics = metric_set(roc_auc, accuracy, sensitivity, specificity, j_index),
#     control = control_resamples(save_pred = TRUE)
#   )
# 
# saveRDS(glm_rs, here("out", "glm_rs.rds"))
glm_rs <- readRDS(here("out", "glm_rs.rds"))

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
# doParallel::registerDoParallel()
# set.seed(1234)
# rf_tune_rs <- tune_grid(
#   credit_wf %>% add_model(rf_spec),
#   resamples = credit_folds,
#   grid = 20
# )
# 
# saveRDS(rf_tune_rs, file = here("out", "rf_tune_rs.rds"))
rf_tune_rs <- readRDS(here("out", "rf_tune_rs.rds"))

rf_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, min_n, mtry) %>%
  pivot_longer(
    cols = min_n:mtry, 
    values_to = "value",
    names_to = "parameter") %>%
  ggplot(aes(x = value, y = mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") + 
  labs(x = NULL, y = "AUC")

#' We can see that lower values of `min_n` are better, and no pattern with `mtry`.
#' Let's create a regular grid to do a finer optimization.

rf_grid <- grid_regular(
  mtry(range = c(0, 25)),
  min_n(range = c(1, 10)),
  levels = 6
)

rf_grid


# doParallel::registerDoParallel()
# set.seed(1234)
# rf_reg_tune_rs <- tune_grid(
#   credit_wf %>% add_model(rf_spec),
#   resamples = credit_folds,
#   grid = rf_grid
# )
# 
# saveRDS(rf_reg_tune_rs, file = here("out", "rf_reg_tune_rs.rds"))
rf_reg_tune_rs <- readRDS(here("out", "rf_reg_tune_rs.rds"))

rf_reg_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(x = mtry, y = mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "AUC")

rf_reg_tune_rs %>%
  collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(x = mtry, y = mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "Accuracy")

#' We can see from the plot of AUC that the best combination is `min_n = 1`, and 
#' `mtry = 10`. There seems to be a decline in accuracy from `mtry = 5`, however,
#' this is likely due to reduced sensitivity and improved specificity, which is
#' the opposite of what we're interested in given the class imbalance.
#' It is generally accepted that good starting points are `mtry = sqrt(p)` (c. 5)
#' and `min_n = 1` for classification models (https://bradleyboehmke.github.io/HOML/random-forest.html)

best_auc <- select_best(rf_reg_tune_rs, "roc_auc")

rf_final_spec <- finalize_model(
  rf_spec,
  best_auc
)

# Fit random forest model to all folds in training data (resampling), saving certain metrics
rf_final_rs <- credit_wf %>%
  add_model(rf_final_spec) %>%
  fit_resamples(
    resamples = credit_folds,
    metrics = metric_set(roc_auc, accuracy, sensitivity, specificity, j_index),
    control = control_resamples(save_pred = TRUE)
  )

rf_final_rs


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
  grid = xgb_grid
)

saveRDS(xgb_tune_rs, file = here("out", "xgb_tune_rs.rds"))
xgb_tune_rs <- readRDS(here("out", "xgb_tune_rs.rds"))

# Evaluate Models ---------------------------------------------------------

collect_metrics(glm_rs)
collect_metrics(rf_final_rs)


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