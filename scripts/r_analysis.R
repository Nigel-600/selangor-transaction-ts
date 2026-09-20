{
  getwd()
  source(file.path(getwd(), "scripts", "packages.R"))
  source(file.path(getwd(), "scripts", "functions.R"))
}

{
  usethis::edit_r_environ(scope="project")
  usethis::edit_r_profile(scope="project")
}

{
  k_rev      <- 7  # number of Revenue lags
  k_disc     <- 7  # number of Discount lags
}
# Fitting Store PCCE
{
  # store panel data, store_plm_dat, store_plm_dat_dropna
  # store_plm_dat : Dropped stores
  {
    store_data <- read.csv("data/panels/store/combined_store_panels_lags.csv")
    store_data$Time_Index <- as.numeric(as.Date(store_data$Sales.Date) - as.Date("2023-01-01")) + 1
    stores_to_remove <- c(31, 38, 39)
    store_data_drop <- store_data %>%
      dplyr::filter(!Store.Code %in% stores_to_remove)
    store_plm_dat <- store_data_drop
  }
  
  
  
  
  # Formula lags and formula itself
  {
    if (k_disc > 0){
      rev_lags  <- paste0("Revenue.lag",  1:k_rev,  collapse = " + ")
      disc_srs <- paste0("Discount", collapse = " + ")
      disc_lags <- paste0("Discount.lag", 1:k_disc, collapse = " + ")
      rhs <- paste(rev_lags, disc_srs, disc_lags, sep = " + ")
      formula <- as.formula(paste("Revenue ~", rhs))
    }
    else{
      rev_lags  <- paste0("Revenue.lag",  1:k_rev,  collapse = " + ")
      disc_srs <- paste0("Discount", collapse = " + ")
      rhs <- paste(rev_lags, disc_srs, sep = " + ")
      formula <- as.formula(paste("Revenue ~", rhs))
    }
  }
  
  # DROPPED-NA input matrix.
  {
    {
      if (k_disc > 0){
        formula_cols <- c(
          "Revenue",
          paste0("Revenue.lag",  1:k_rev),
          "Discount",
          paste0("Discount.lag", 1:k_disc)
        )
      }
      else{
        formula_cols <- c(
          "Revenue",
          paste0("Revenue.lag",  1:k_rev),
          "Discount"
        )
      }
      
      index_cols <- c("Store.Code", "Time_Index")
      keep_cols  <- c(index_cols, formula_cols)
    }
    
    store_plm_dat_dropna <- store_plm_dat[complete.cases(store_plm_dat[, keep_cols]), ]
  }
  
  
  store_ccemg <- pcce(formula, data = store_plm_dat, index = c("Store.Code", "Time_Index"), model = "mg")
  store_vcm <- pvcm(formula, data = store_plm_dat, index = c("Store.Code", "Time_Index"), model = "within")
  
  # Store res_df
  {
    res_df <- store_plm_dat_dropna
    for (i in 1:k_rev){
      for (j in unique(res_df$Store.Code)){
        res_df[["Revenue"]][which(res_df$Store.Code == j)] <- store_ccemg$residuals[which(res_df$Store.Code == j)]
      }
      
      if (paste0("Revenue.lag", i) %in% names(res_df)) res_df[[paste0("Revenue.lag", i)]] <- NULL
      if (paste0("Discount.lag", i) %in% names(res_df)) res_df[[paste0("Discount.lag", i)]] <- NULL
    }
    res_df$Time_Index <- NULL
    
    
    write.csv(res_df, "data/residuals_store_panel.csv", row.names = FALSE)
  }
  
  # Maddala-wu
  {
    madwu_data <- store_plm_dat_dropna
    madwu_data$Revenue <- store_ccemg$residuals
    madwu_data <- pdata.frame(madwu_data, index = c("Store.Code", "Time_Index"))
    mwu_test <- purtest(
      madwu_data$Revenue,
      test = "madwu",
      lags = "AIC"
    )
    for (store_code in unique(madwu_data$Store.Code)){
      # Use double brackets [[ ]] for direct access, and convert to character
      lags_value <- mwu_test$idres[[as.character(store_code)]]$lags
      print(paste("Store", store_code, "requires", lags_value, "lags."))
    }
  }
  
  summary(store_ccemg, vcov = function(x) vcovSCC(x, type = "HC3", inner = "cluster"))
  
}

{
  pgranger_data <- store_plm_dat_dropna
  pgranger_data$Revenue <- store_ccemg$tr.model$y
  pgranger_data$Discount <- store_ccemg$tr.model$X[, "Discount"]
  pgt <- pgrangertest(
    formula = Revenue ~ Discount,
    data = pgranger_data,
    order = 1,
    test = "Ztilde",
    index = c("Store.Code", "Time_Index")
  )
  
  {
    library(ggplot2)
    
    # Extract the data from pgt result (keeping original order)
    plot_data <- data.frame(
      Store.Code = pgt$indgranger$Store.Code,
      p_value = pgt$indgranger$`p-value`
    )
    
    # Create a column to identify significant vs non-significant
    plot_data$sig <- ifelse(plot_data$p_value < 0.05, "Below 0.05", "Above 0.05")
    
    # Create ggplot with store codes on top of bars
    ggplot(plot_data, aes(x = factor(Store.Code, levels = Store.Code), 
                          y = p_value,
                          fill = sig)) +
      geom_bar(stat = "identity", show.legend = FALSE) +
      geom_text(aes(label = Store.Code), 
                vjust = -0.5,  # Position above the bar
                size = 3) +
      geom_hline(yintercept = 0.05, linetype = 6, linewidth = 1, color = "red") +
      scale_fill_manual(values = c("Below 0.05" = "lightgray", 
                                   "Above 0.05" = "steelblue")) +
      scale_y_continuous(breaks = c(0, 0.05, 0.1, 0.15, 0.2)) +  # Force a tick at 0.05
      labs(x = "",  # No x-axis label since codes are on bars
           y = "p-value") +
      theme_minimal() +
      theme(axis.text.x = element_blank(),  # Remove x-axis text
            axis.ticks.x = element_blank())  # Remove x-axis ticks
  }
  print(pgt)
}
# vcm qqplot
{
  file_path <- paste0("figs/resids/", k_rev, "/store_vcm_residual_plots.pdf")
  dir.create(dirname(file_path), showWarnings = FALSE, recursive = TRUE)
  pdf(file_path)
  
  store_codes <- unique(store_plm_dat_dropna$Store.Code)
  
  for (store_code in store_codes) {
    par(mfrow = c(2, 1))
    
    idx <- which(store_plm_dat_dropna$Store.Code == store_code)
    res <- store_vcm$residuals[idx]
    
    # QQ plot
    qqnorm(res, main = paste("QQ Plot - Store", store_code))
    qqline(res)
    
    # Residual time plot
    plot(res, main = paste("Residuals - Store", store_code),
         ylab = "Residuals", xlab = "Index")
  }
  
  dev.off()
}

  # ccemg qqplot
{
  file_path <- paste0("figs/resids/", k_rev, "/store_ccemg_residual_plots.pdf")
  dir.create(dirname(file_path), showWarnings = FALSE, recursive = TRUE)
  pdf(file_path)
  
  store_codes <- unique(store_plm_dat_dropna$Store.Code)
  
  for (store_code in store_codes) {
    par(mfrow = c(2, 1))
    
    idx <- which(store_plm_dat_dropna$Store.Code == store_code)
    res <- store_ccemg$residuals[idx]
    
    # QQ plot
    qqnorm(res, main = paste("QQ Plot - Store", store_code))
    qqline(res)
    
    # Residual time plot
    plot(res, main = paste("Residuals - Store", store_code),
         ylab = "Residuals", xlab = "Index")
  }
  
  dev.off()
}





{
  for (orders in 1:7){
    pgranger_data <- store_plm_dat
    pgt <- pgrangertest(
      formula = Revenue ~ Discount,
      data = pgranger_data,
      order = orders,
      test = "Ztilde",
      index = c("Store.Code", "Time_Index")
    )
    print(pgt$p.value)
  }
  
  {
    library(ggplot2)
    
    # Extract the data from pgt result (keeping original order)
    plot_data <- data.frame(
      Store.Code = pgt$indgranger$Store.Code,
      p_value = pgt$indgranger$`p-value`
    )
    
    # Create a column to identify significant vs non-significant
    plot_data$sig <- ifelse(plot_data$p_value < 0.05, "Below 0.05", "Above 0.05")
    
    # Create ggplot with store codes on top of bars
    ggplot(plot_data, aes(x = factor(Store.Code, levels = Store.Code), 
                          y = p_value,
                          fill = sig)) +
      geom_bar(stat = "identity", show.legend = FALSE) +
      geom_text(aes(label = Store.Code), 
                vjust = -0.5,  # Position above the bar
                size = 3) +
      geom_hline(yintercept = 0.05, linetype = 6, linewidth = 1, color = "red") +
      scale_fill_manual(values = c("Below 0.05" = "lightgray", 
                                   "Above 0.05" = "steelblue")) +
      scale_y_continuous(breaks = c(0, 0.05, 0.1, 0.15, 0.2)) +  # Force a tick at 0.05
      labs(x = "",  # No x-axis label since codes are on bars
           y = "p-value") +
      theme_minimal() +
      theme(axis.text.x = element_blank(),  # Remove x-axis text
            axis.ticks.x = element_blank())  # Remove x-axis ticks
  }
  # print(pgt)
  # print(pgt$indgranger)
}



{
  pgranger_data <- store_plm_dat_dropna
  pgranger_data$Revenue <- store_ccemg$tr.model$y
  pgranger_data$Discount <- store_ccemg$tr.model$X[, "Discount"]
  pgt <- pgrangertest(
    formula = Revenue ~ Discount,
    data = pgranger_data,
    order = 1,
    test = "Wbar",
    index = c("Store.Code", "Time_Index")
  )
  
  {
    library(ggplot2)
    
    # Extract the data from pgt result (keeping original order)
    plot_data <- data.frame(
      Store.Code = pgt$indgranger$Store.Code,
      p_value = pgt$indgranger$`p-value`
    )
    
    # Create a column to identify significant vs non-significant
    plot_data$sig <- ifelse(plot_data$p_value < 0.05, "Below 0.05", "Above 0.05")
    
    # Create ggplot with store codes on top of bars
    ggplot(plot_data, aes(x = factor(Store.Code, levels = Store.Code), 
                          y = p_value,
                          fill = sig)) +
      geom_bar(stat = "identity", show.legend = FALSE) +
      geom_text(aes(label = Store.Code), 
                vjust = -0.5,  # Position above the bar
                size = 3) +
      geom_hline(yintercept = 0.05, linetype = 6, linewidth = 1, color = "red") +
      scale_fill_manual(values = c("Below 0.05" = "lightgray", 
                                   "Above 0.05" = "steelblue")) +
      scale_y_continuous(breaks = c(0, 0.05, 0.1, 0.15, 0.2)) +  # Force a tick at 0.05
      labs(x = "",  # No x-axis label since codes are on bars
           y = "p-value") +
      theme_minimal() +
      theme(axis.text.x = element_blank(),  # Remove x-axis text
            axis.ticks.x = element_blank())  # Remove x-axis ticks
  }
  
}

# Mundlak (store)
{
  re_store <- pvcm(
    formula = formula,
    data = store_plm_dat,
    index = c("Store.Code", "Time_Index"),
    model = "random",
    effect = "individual"
  )
  mundlak_dat <- store_plm_dat %>%
    group_by(Store.Code) %>%
    mutate(
      Revenue_mean       = mean(Revenue,       na.rm = TRUE),
      Revenue.lag1_mean  = mean(Revenue.lag1,  na.rm = TRUE),
      Revenue.lag2_mean  = mean(Revenue.lag2,  na.rm = TRUE),
      Revenue.lag3_mean  = mean(Revenue.lag3,  na.rm = TRUE),
      Revenue.lag4_mean  = mean(Revenue.lag4,  na.rm = TRUE),
      Revenue.lag5_mean  = mean(Revenue.lag5,  na.rm = TRUE),
      Revenue.lag6_mean  = mean(Revenue.lag6,  na.rm = TRUE),
      Revenue.lag7_mean  = mean(Revenue.lag7,  na.rm = TRUE),
      Discount_mean      = mean(Discount,      na.rm = TRUE),
      Discount.lag1_mean = mean(Discount.lag1, na.rm = TRUE),
      Discount.lag2_mean = mean(Discount.lag2, na.rm = TRUE),
      Discount.lag3_mean = mean(Discount.lag3, na.rm = TRUE),
      Discount.lag4_mean = mean(Discount.lag4, na.rm = TRUE),
      Discount.lag5_mean = mean(Discount.lag5, na.rm = TRUE),
      Discount.lag6_mean = mean(Discount.lag6, na.rm = TRUE),
      Discount.lag7_mean = mean(Discount.lag7, na.rm = TRUE)
    ) %>%
    ungroup()
  # rX_means, dX_means
  {
    r1_means <- numeric(length(unique(mundlak_dat$Store.Code)))
    r2_means <- numeric(length(unique(mundlak_dat$Store.Code)))
    r3_means <- numeric(length(unique(mundlak_dat$Store.Code)))
    r4_means <- numeric(length(unique(mundlak_dat$Store.Code)))
    r5_means <- numeric(length(unique(mundlak_dat$Store.Code)))
    r6_means <- numeric(length(unique(mundlak_dat$Store.Code)))
    r7_means <- numeric(length(unique(mundlak_dat$Store.Code)))
    d0_means <- numeric(length(unique(mundlak_dat$Store.Code)))
    d1_means <- numeric(length(unique(mundlak_dat$Store.Code)))
    d2_means <- numeric(length(unique(mundlak_dat$Store.Code)))
    d3_means <- numeric(length(unique(mundlak_dat$Store.Code)))
    d4_means <- numeric(length(unique(mundlak_dat$Store.Code)))
    d5_means <- numeric(length(unique(mundlak_dat$Store.Code)))
    d6_means <- numeric(length(unique(mundlak_dat$Store.Code)))
    d7_means <- numeric(length(unique(mundlak_dat$Store.Code)))
    }
  
  
  i <- 1
  for (store_code in unique(mundlak_dat$Store.Code)){
    r1_means[i] <- mundlak_dat$Revenue.lag1_mean[which(mundlak_dat$Store.Code == store_code)][1]
    r2_means[i] <- mundlak_dat$Revenue.lag2_mean[which(mundlak_dat$Store.Code == store_code)][1]
    r3_means[i] <- mundlak_dat$Revenue.lag3_mean[which(mundlak_dat$Store.Code == store_code)][1]
    r4_means[i] <- mundlak_dat$Revenue.lag4_mean[which(mundlak_dat$Store.Code == store_code)][1]
    r5_means[i] <- mundlak_dat$Revenue.lag5_mean[which(mundlak_dat$Store.Code == store_code)][1]
    r6_means[i] <- mundlak_dat$Revenue.lag6_mean[which(mundlak_dat$Store.Code == store_code)][1]
    r7_means[i] <- mundlak_dat$Revenue.lag7_mean[which(mundlak_dat$Store.Code == store_code)][1]
    d0_means[i] <- mundlak_dat$Discount_mean[which(mundlak_dat$Store.Code == store_code)][1]
    d1_means[i] <- mundlak_dat$Discount.lag1_mean[which(mundlak_dat$Store.Code == store_code)][1]
    d2_means[i] <- mundlak_dat$Discount.lag2_mean[which(mundlak_dat$Store.Code == store_code)][1]
    d3_means[i] <- mundlak_dat$Discount.lag3_mean[which(mundlak_dat$Store.Code == store_code)][1]
    d4_means[i] <- mundlak_dat$Discount.lag4_mean[which(mundlak_dat$Store.Code == store_code)][1]
    d5_means[i] <- mundlak_dat$Discount.lag5_mean[which(mundlak_dat$Store.Code == store_code)][1]
    d6_means[i] <- mundlak_dat$Discount.lag6_mean[which(mundlak_dat$Store.Code == store_code)][1]
    d7_means[i] <- mundlak_dat$Discount.lag7_mean[which(mundlak_dat$Store.Code == store_code)][1]
    i <- i + 1
  }
  
  mund_aux_data <- data.frame(
    alpha = re_store$single.coefs[,1],
    r1 = r1_means,
    r2 = r2_means,
    r3 = r3_means,
    r4 = r4_means,
    r5 = r5_means,
    r6 = r6_means,
    r7 = r7_means,
    d0 = d0_means,
    d1 = d1_means,
    d2 = d2_means,
    d3 = d3_means,
    d4 = d4_means,
    d5 = d5_means,
    d6 = d6_means,
    d7 = d7_means
  )
  
  aux_model <- lm(
    formula = alpha ~ . - 1,
    data = mund_aux_data
  )
  summary(aux_model)
}

# drink panel data, drink_plm_dat, drink_plm_dat_dropna
{
  drink_data <- read.csv("data/panels/drink/combined_drink_panel_lags.csv")
  drink_data$Time_Index <- as.numeric(as.Date(drink_data$Sales.Date) - as.Date("2023-01-01")) + 1
  drink_plm_dat <- drink_data
}

# Mundlak (drink)
{
  re_drink <- pvcm(
    formula = formula,
    data = drink_plm_dat,
    index = c("Product.Name", "Time_Index"),
    model = "random",
    effect = "individual"
  )
  mundlak_dat <- drink_plm_dat %>%
    group_by(Product.Name) %>%
    mutate(
      Revenue_mean       = mean(Revenue,       na.rm = TRUE),
      Revenue.lag1_mean  = mean(Revenue.lag1,  na.rm = TRUE),
      Revenue.lag2_mean  = mean(Revenue.lag2,  na.rm = TRUE),
      Revenue.lag3_mean  = mean(Revenue.lag3,  na.rm = TRUE),
      Revenue.lag4_mean  = mean(Revenue.lag4,  na.rm = TRUE),
      Revenue.lag5_mean  = mean(Revenue.lag5,  na.rm = TRUE),
      Revenue.lag6_mean  = mean(Revenue.lag6,  na.rm = TRUE),
      Revenue.lag7_mean  = mean(Revenue.lag7,  na.rm = TRUE),
      Discount_mean      = mean(Discount,      na.rm = TRUE),
      Discount.lag1_mean = mean(Discount.lag1, na.rm = TRUE),
      Discount.lag2_mean = mean(Discount.lag2, na.rm = TRUE),
      Discount.lag3_mean = mean(Discount.lag3, na.rm = TRUE),
      Discount.lag4_mean = mean(Discount.lag4, na.rm = TRUE),
      Discount.lag5_mean = mean(Discount.lag5, na.rm = TRUE),
      Discount.lag6_mean = mean(Discount.lag6, na.rm = TRUE),
      Discount.lag7_mean = mean(Discount.lag7, na.rm = TRUE)
    ) %>%
    ungroup()
  # rX_means, dX_means
  {
    r1_means <- numeric(length(unique(mundlak_dat$Product.Name)))
    r2_means <- numeric(length(unique(mundlak_dat$Product.Name)))
    r3_means <- numeric(length(unique(mundlak_dat$Product.Name)))
    r4_means <- numeric(length(unique(mundlak_dat$Product.Name)))
    r5_means <- numeric(length(unique(mundlak_dat$Product.Name)))
    r6_means <- numeric(length(unique(mundlak_dat$Product.Name)))
    r7_means <- numeric(length(unique(mundlak_dat$Product.Name)))
    d0_means <- numeric(length(unique(mundlak_dat$Product.Name)))
    d1_means <- numeric(length(unique(mundlak_dat$Product.Name)))
    d2_means <- numeric(length(unique(mundlak_dat$Product.Name)))
    d3_means <- numeric(length(unique(mundlak_dat$Product.Name)))
    d4_means <- numeric(length(unique(mundlak_dat$Product.Name)))
    d5_means <- numeric(length(unique(mundlak_dat$Product.Name)))
    d6_means <- numeric(length(unique(mundlak_dat$Product.Name)))
    d7_means <- numeric(length(unique(mundlak_dat$Product.Name)))
    }
  
  i <- 1
  for (product_name in unique(mundlak_dat$Product.Name)){
    r1_means[i] <- mundlak_dat$Revenue.lag1_mean[which(mundlak_dat$Product.Name == product_name)][1]
    r2_means[i] <- mundlak_dat$Revenue.lag2_mean[which(mundlak_dat$Product.Name == product_name)][1]
    r3_means[i] <- mundlak_dat$Revenue.lag3_mean[which(mundlak_dat$Product.Name == product_name)][1]
    r4_means[i] <- mundlak_dat$Revenue.lag4_mean[which(mundlak_dat$Product.Name == product_name)][1]
    r5_means[i] <- mundlak_dat$Revenue.lag5_mean[which(mundlak_dat$Product.Name == product_name)][1]
    r6_means[i] <- mundlak_dat$Revenue.lag6_mean[which(mundlak_dat$Product.Name == product_name)][1]
    r7_means[i] <- mundlak_dat$Revenue.lag7_mean[which(mundlak_dat$Product.Name == product_name)][1]
    d0_means[i] <- mundlak_dat$Discount_mean[which(mundlak_dat$Product.Name == product_name)][1]
    d1_means[i] <- mundlak_dat$Discount.lag1_mean[which(mundlak_dat$Product.Name == product_name)][1]
    d2_means[i] <- mundlak_dat$Discount.lag2_mean[which(mundlak_dat$Product.Name == product_name)][1]
    d3_means[i] <- mundlak_dat$Discount.lag3_mean[which(mundlak_dat$Product.Name == product_name)][1]
    d4_means[i] <- mundlak_dat$Discount.lag4_mean[which(mundlak_dat$Product.Name == product_name)][1]
    d5_means[i] <- mundlak_dat$Discount.lag5_mean[which(mundlak_dat$Product.Name == product_name)][1]
    d6_means[i] <- mundlak_dat$Discount.lag6_mean[which(mundlak_dat$Product.Name == product_name)][1]
    d7_means[i] <- mundlak_dat$Discount.lag7_mean[which(mundlak_dat$Product.Name == product_name)][1]
    i <- i + 1
  }
  
  mund_aux_data <- data.frame(
    alpha = re_store$single.coefs[,1],
    r1 = r1_means,
    r2 = r2_means,
    r3 = r3_means,
    r4 = r4_means,
    r5 = r5_means,
    r6 = r6_means,
    r7 = r7_means,
    d0 = d0_means,
    d1 = d1_means,
    d2 = d2_means,
    d3 = d3_means,
    d4 = d4_means,
    d5 = d5_means,
    d6 = d6_means,
    d7 = d7_means
  )
  
  aux_model <- lm(
    formula = alpha ~ . - 1,
    data = mund_aux_data
  )
  summary(aux_model)
}


k_rev      <- 7  # number of Revenue lags
k_disc     <- 7 # number of Discount lags
# Fitting Drink PCCE
{
  # drink panel data, drink_plm_dat, drink_plm_dat_dropna
  {
    drink_data <- read.csv("data/panels/drink/combined_drink_panel_lags.csv")
    drink_data$Time_Index <- as.numeric(as.Date(drink_data$Sales.Date) - as.Date("2023-01-01")) + 1
    drink_plm_dat <- drink_data
  }
  
  
  # Formula lags
  {
    if (k_disc > 0){
      rev_lags  <- paste0("Revenue.lag",  1:k_rev,  collapse = " + ")
      disc_srs <- paste0("Discount", collapse = " + ")
      disc_lags <- paste0("Discount.lag", 1:k_disc, collapse = " + ")
      rhs <- paste(rev_lags, disc_srs, disc_lags, sep = " + ")
      formula <- as.formula(paste("Revenue ~", rhs))
    }
    else{
      rev_lags  <- paste0("Revenue.lag",  1:k_rev,  collapse = " + ")
      disc_srs <- paste0("Discount", collapse = " + ")
      rhs <- paste(rev_lags, disc_srs, sep = " + ")
      formula <- as.formula(paste("Revenue ~", rhs))
    }
  }
  
  # Formula itself
  {
    if (k_disc > 0){
      formula_cols <- c(
        "Revenue",
        paste0("Revenue.lag",  1:k_rev),
        "Discount",
        paste0("Discount.lag", 1:k_disc)
      )
    }
    else{
      formula_cols <- c(
        "Revenue",
        paste0("Revenue.lag",  1:k_rev),
        "Discount"
      )
    }
    
    index_cols <- c("Product.Name", "Time_Index")
    keep_cols  <- c(index_cols, formula_cols)
  }
  drink_plm_dat_dropna <- drink_plm_dat[complete.cases(drink_plm_dat[, keep_cols]), ]
  
  
  
  
  drink_ccemg <- pcce(formula, data = drink_plm_dat, index = c("Product.Name", "Time_Index"), model = "mg")
  drink_vcm <- pvcm(formula, data = drink_plm_dat, index = c("Product.Name", "Time_Index"), model = "within")
  # Maddala-wu
  {
    madwu_data <- drink_plm_dat_dropna
    madwu_data$Revenue <- drink_ccemg$residuals
    madwu_data <- pdata.frame(madwu_data, index = c("Product.Name", "Time_Index"))
    mwu_test <- purtest(
      madwu_data$Revenue,
      test = "madwu",
      lags = "AIC"
    )
    for (prod_name in unique(madwu_data$Product.Name)){
      # Use double brackets [[ ]] for direct access, and convert to character
      lags_value <- mwu_test$idres[[as.character(prod_name)]]$lags
      print(paste("Drink", prod_name, "requires", lags_value, "lags."))
    }
  }
  
  # Store res_df
  {
    res_df <- drink_plm_dat_dropna
    for (i in 1:k_rev){
      for (j in unique(res_df$Product.Name)){
        res_df[["Revenue"]][which(res_df$Product.Name == j)] <- drink_ccemg$residuals[which(res_df$Product.Name == j)]
      }
      
      if (paste0("Revenue.lag", i) %in% names(res_df)) res_df[[paste0("Revenue.lag", i)]] <- NULL
      if (paste0("Discount.lag", i) %in% names(res_df)) res_df[[paste0("Discount.lag", i)]] <- NULL
    }
    res_df$Time_Index <- NULL
    
    
    write.csv(res_df, "data/residuals_drink_panel.csv", row.names = FALSE)
  }
  summary(drink_ccemg, vcov = function(x) vcovSCC(x, inner = "cluster", type = "HC3"))
}



{
  #drink data
  {
    drink_data <- read.csv("data/panels/drink/combined_drink_panel_lags.csv")
    drink_data$Time_Index <- as.numeric(as.Date(drink_data$Sales.Date) - as.Date("2023-01-01")) + 1
    drink_plm_dat <- drink_data
  }
  
  pgranger_data <- drink_plm_dat_dropna
  for (orders in 1:7){
    k_rev      <- orders  # number of Revenue lags
    k_disc     <- k_rev   # number of Discount lags
    #get formula
    {
      # Formula lags
      {
        rev_lags  <- paste0("Revenue.lag",  1:k_rev,  collapse = " + ")
        disc_srs <- paste0("Discount", collapse = " + ")
        disc_lags <- paste0("Discount.lag", 1:k_disc, collapse = " + ")
      }
      
      # Formula itself
      rhs <- paste(rev_lags, disc_lags, sep = " + ")
      formula <- as.formula(paste("Revenue ~", rhs))
    }
    
    
    # DROPPED-NA input matrix. (pgranger data)
    {
      formula_cols <- c(
        "Revenue",
        paste0("Revenue.lag",  1:k_rev),
        "Discount",
        paste0("Discount.lag", 1:k_disc)
      )
      index_cols <- c("Product.Name", "Time_Index")
      keep_cols  <- c(index_cols, formula_cols)
      
      drink_plm_dat_dropna <- drink_plm_dat[complete.cases(drink_plm_dat[, keep_cols]), ]
    }
    
    
    
    pgt <- pgrangertest(
      formula = Revenue ~ Discount,
      data = pgranger_data,
      order = orders,
      test = "Ztilde",
      index = c("Product.Name", "Time_Index")
    )
    
    print(paste0("Lag Order : ", orders))
    print(pgt$p.value)
    
    drink_vcm <- pvcm(formula, data = drink_plm_dat, 
                      index = c("Product.Name", "Time_Index"), 
                      model = "within")
    
    
    print(pcdtest(drink_vcm, "rho"))
    print(pcdtest(drink_vcm, "absrho"))
    
    
    
  }
  
  # print(pgt)
  # print(pgt$indgranger)
}

{
  #drink data
  {
    {
      drink_data <- read.csv("data/panels/drink/combined_drink_panel_lags.csv")
      drink_data$Time_Index <- as.numeric(as.Date(drink_data$Sales.Date) - as.Date("2023-01-01")) + 1
      drink_plm_dat <- drink_data
    }
  }
  
  
  k_rev      <- 7  # number of Revenue lags
  k_disc     <- 1   # number of Discount lags
  #get formula
  {
    {
      # Formula lags
      {
        rev_lags  <- paste0("Revenue.lag",  1:k_rev,  collapse = " + ")
        disc_srs <- paste0("Discount", collapse = " + ")
        disc_lags <- paste0("Discount.lag", 1:k_disc, collapse = " + ")
      }
      
      # Formula itself
      rhs <- paste(rev_lags, disc_srs, disc_lags, sep = " + ")
      formula <- as.formula(paste("Revenue ~", rhs))
    }
  }
  {
    # DROPPED-NA input matrix. (pgranger data)
    {
      {
        formula_cols <- c(
          "Revenue",
          paste0("Revenue.lag",  1:k_rev),
          "Discount",
          paste0("Discount.lag", 1:k_disc)
        )
        index_cols <- c("Product.Name", "Time_Index")
        keep_cols  <- c(index_cols, formula_cols)
        
        drink_plm_dat_dropna <- drink_plm_dat[complete.cases(drink_plm_dat[, keep_cols]), ]
      }
    }
  }
  
  
  
  {
    # pgranger data
    {
      pgranger_data <- drink_plm_dat_dropna
      drink_ccemg_granger <- pcce(formula, data = drink_plm_dat, index = c("Product.Name", "Time_Index"), model = "mg")
      pgranger_data$Revenue <- drink_ccemg_granger$tr.model$y
      pgranger_data$Discount <- drink_ccemg_granger$tr.model$X[, "Discount"]
    }
  }
  
  for (orders in 1:7){
    k_rev      <- orders  # number of Revenue lags
    k_disc     <- k_rev   # number of Discount lags
    
    
    
    
    
    pgt <- pgrangertest(
      formula = Revenue ~ Discount,
      data = pgranger_data,
      order = orders,
      test = "Ztilde",
      index = c("Product.Name", "Time_Index")
    )
    
    print(paste0("Lag Order : ", orders))
    print(pgt$p.value)
    #get formula
    {
      {
        # Formula lags
        {
          rev_lags  <- paste0("Revenue.lag",  1:orders,  collapse = " + ")
          disc_srs <- paste0("Discount", collapse = " + ")
          disc_lags <- paste0("Discount.lag", 1:orders, collapse = " + ")
        }
        
        # Formula itself
        rhs <- paste(rev_lags, disc_lags, sep = " + ")
        formula <- as.formula(paste("Revenue ~", rhs))
      }
    }
    drink_vcm <- pvcm(formula, data = pgranger_data, 
                      index = c("Product.Name", "Time_Index"), 
                      model = "within")
    
    
    print(pcdtest(drink_vcm, "rho"))
    print(pcdtest(drink_vcm, "absrho"))
    
    
    
  }
  
  # print(pgt)
  # print(pgt$indgranger)
}




{
  pgranger_data <- drink_plm_dat_dropna
  pgranger_data$Revenue <- drink_ccemg$tr.model$y
  pgranger_data$Discount <- drink_ccemg$tr.model$X[, "Discount"]
  for (orders in 1:7){
    pgt <- pgrangertest(
      formula = Revenue ~ Discount,
      data = pgranger_data,
      order = orders,
      test = "Wbar",
      index = c("Product.Name", "Time_Index")
    )  
  }
  
  print(paste0("Lag Order : ", orders))
  print(pgt$p.value)
  print(pcdtest(drink_vcm, "rho"))
  print(pcdtest(drink_vcm, "absrho"))
  {
    
    # Extract the data from pgt result (keeping original order)
    plot_data <- data.frame(
      Product.Name = pgt$indgranger$Product.Name,
      p_value = pgt$indgranger$`p-value`
    )
    
    # Create a column to identify significant vs non-significant
    plot_data$sig <- ifelse(plot_data$p_value < 0.05, "Below 0.05", "Above 0.05")
    
    # Create ggplot with store codes on top of bars
    ggplot(plot_data, aes(x = factor(Product.Name, levels = Product.Name), 
                          y = p_value,
                          fill = sig)) +
      geom_bar(stat = "identity", show.legend = FALSE) +
      geom_text(aes(label = Product.Name), 
                vjust = -0.5,  # Position above the bar
                size = 3) +
      geom_hline(yintercept = 0.05, linetype = 6, linewidth = 1, color = "red") +
      scale_fill_manual(values = c("Below 0.05" = "lightgray", 
                                   "Above 0.05" = "steelblue")) +
      scale_y_continuous(breaks = c(0, 0.05, 0.1, 0.15, 0.2)) +  # Force a tick at 0.05
      labs(x = "",  # No x-axis label since codes are on bars
           y = "p-value") +
      theme_minimal() +
      theme(axis.text.x = element_blank(),  # Remove x-axis text
            axis.ticks.x = element_blank())  # Remove x-axis ticks
  }
}

{
  pgranger_data <- drink_plm_dat
  pgt <- pgrangertest(
    formula = Revenue ~ Discount,
    data = pgranger_data,
    order = 7,
    test = "Wbar",
    index = c("Product.Name", "Time_Index")
  )
  
  {
    
    # Extract the data from pgt result (keeping original order)
    plot_data <- data.frame(
      Product.Name = pgt$indgranger$Product.Name,
      p_value = pgt$indgranger$`p-value`
    )
    
    # Create a column to identify significant vs non-significant
    plot_data$sig <- ifelse(plot_data$p_value < 0.05, "Below 0.05", "Above 0.05")
    
    # Create ggplot with store codes on top of bars
    ggplot(plot_data, aes(x = factor(Product.Name, levels = Product.Name), 
                          y = p_value,
                          fill = sig)) +
      geom_bar(stat = "identity", show.legend = FALSE) +
      geom_text(aes(label = Product.Name), 
                vjust = -0.5,  # Position above the bar
                size = 3) +
      geom_hline(yintercept = 0.05, linetype = 6, linewidth = 1, color = "red") +
      scale_fill_manual(values = c("Below 0.05" = "lightgray", 
                                   "Above 0.05" = "steelblue")) +
      scale_y_continuous(breaks = c(0, 0.05, 0.1, 0.15, 0.2)) +  # Force a tick at 0.05
      labs(x = "",  # No x-axis label since codes are on bars
           y = "p-value") +
      theme_minimal() +
      theme(axis.text.x = element_blank(),  # Remove x-axis text
            axis.ticks.x = element_blank())  # Remove x-axis ticks
  }
  # print(pgt)
}


# Plotting Dark Choco Latte
{
  rev_per_store <- read.csv("data/per_store/revenue_per_store.csv")
  rev_per_drink <- read.csv("data/per_drink/revenue_per_drink.csv")
  
  disc_per_store <- read.csv("data/per_store/discount_per_store.csv")
  disc_per_drink <- read.csv("data/per_drink/discount_per_drink.csv")
  
}






















{
  drink_name <- "Total"
  bp_df <- rev_per_drink
  bp_df$time <- time(rev_per_drink$Total)
  result <- analyze_breaks(bp_df, drink_name, h = 0.10)
  total_rev_breakpoints <- result$bp$breakpoints
}



{
  data_at_drink_bp <- rev_per_drink[total_rev_breakpoints, ]
}

{
  data_at_unit_price <- rev_per_drink[which(rev_per_drink$Sales.Date == "2023-05-27"), ]
}



{
  p <- plot_revenue_range(rev_per_store, "2023-03-01", "2023-06-30")
  p + geom_hline(yintercept = rev_per_store$Total[which(rev_per_store$Sales.Date == "2023-03-21")], colour = "red", linetype = "dashed" )
}

{
  # 1. Get residuals from your plm model
  resids <- residuals(store_ccemg)
  
  # 2. Add them back to your data frame to keep track of Store and Time
  store_plm_dat_dropna$res <- as.vector(resids)
  
  # 3. Reshape to wide format (Time in rows, Stores in columns)
  library(tidyr)
  wide_res <- pivot_wider(store_plm_dat_dropna, id_cols = Time_Index, names_from = Store.Code, values_from = res)
  
  # 4. Calculate the correlation matrix between stores
  cor_mat <- cor(wide_res[, -1], use = "pairwise.complete.obs")
}

{
  melted <- melt(abs(cor_mat))
  melted$Var1 <- factor(melted$Var1, levels = rownames(cor_mat))
  melted$Var2 <- factor(melted$Var2, levels = colnames(cor_mat))
  
  # Keep only lower triangle (including diagonal)
  melted <- melted[as.integer(melted$Var1) <= as.integer(melted$Var2), ]
  
  ggplot(melted, aes(Var2, Var1, fill = value)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_text(aes(label = round(value, 2)), size = 3, color = "black") +
    scale_fill_gradient2(
      low      = "blue",
      mid      = "white",
      high     = "red",
      midpoint = 0,
      limits   = c(0, 1),
      name     = "r"
    ) +
    scale_x_discrete(drop = FALSE) +
    scale_y_discrete(drop = FALSE) +
    labs(title = "Pairwise Correlation Coefficient Heatmap (CCEMG)") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
      axis.text.y = element_text(size = 9),
      axis.title  = element_blank()
    )
  
}


{
  # 1. Get residuals from your plm model
  resids <- residuals(store_vcm)
  
  # 2. Add them back to your data frame to keep track of Store and Time
  store_plm_dat_dropna$res <- as.vector(resids)
  
  # 3. Reshape to wide format (Time in rows, Stores in columns)
  library(tidyr)
  wide_res <- pivot_wider(store_plm_dat_dropna, id_cols = Time_Index, names_from = Store.Code, values_from = res)
  
  # 4. Calculate the correlation matrix between stores
  cor_mat <- cor(wide_res[, -1], use = "pairwise.complete.obs")
}

{
  
  melted <- melt(abs(cor_mat))
  melted$Var1 <- factor(melted$Var1, levels = rownames(cor_mat))
  melted$Var2 <- factor(melted$Var2, levels = colnames(cor_mat))
  
  # Keep only lower triangle (including diagonal)
  melted <- melted[as.integer(melted$Var1) <= as.integer(melted$Var2), ]
  
  ggplot(melted, aes(Var2, Var1, fill = value)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_text(aes(label = round(value, 2)), size = 3, color = "black") +
    scale_fill_gradient2(
      low      = "blue",
      mid      = "white",
      high     = "red",
      midpoint = 0,
      limits   = c(-1, 1),
      name     = "r"
    ) +
    scale_x_discrete(drop = FALSE) +
    scale_y_discrete(drop = FALSE) +
    labs(title = "Pairwise Correlation Coefficient Heatmap (VCM)") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
      axis.text.y = element_text(size = 9),
      axis.title  = element_blank()
    )
  
}




