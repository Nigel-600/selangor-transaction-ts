{
  usethis::edit_r_environ(scope="project")
  usethis::edit_r_profile(scope="project")
}

{
  analyze_breaks <- function(data, var_name, h = 0.10) {
    
    # Ensure Sales.Date is Date type
    data$Sales.Date <- as.Date(data$Sales.Date)
    
    # Create formulas (still use numeric time for estimation)
    bp_formula <- as.formula(paste(var_name, "~ time"))
    model_formula <- as.formula(paste(var_name, "~ time * breakfactor(bp_model)"))
    
    # Breakpoint analysis
    bp_model <- breakpoints(formula = bp_formula, data = data, h = h)
    print(summary(bp_model))
    
    # Fit model
    model_auto <- lm(model_formula, data = data)
    data$fitted <- fitted(model_auto)
    
    # Map breakpoint indices → actual dates
    break_pts <- bp_model$breakpoints
    break_dates <- data$Sales.Date[break_pts]
    break_df <- data.frame(Sales.Date = break_dates)
    
    # Plot
    p <- ggplot(data, aes(x = Sales.Date, y = .data[[var_name]] / 100000)) +
      geom_line(color = "#2C3E50", linewidth = 1) +
      
      # Fitted line
      geom_line(aes(y = fitted / 100000), color = "#E74C3C", linewidth = 1.2) +
      
      # Breakpoints
      geom_vline(data = break_df,
                 aes(xintercept = Sales.Date),
                 linetype = "dashed",
                 color = "#E74C3C",
                 linewidth = 1) +
      
      # Labels
      labs(
        title = paste("Structural Breaks:", paste0(var_name, " Revenue")),
        subtitle = "Bai-Perron Breakpoints",
        x = "Time",
        y = paste0(var_name, " Revenue (Millions)")
      ) +
      
      # Monthly gridlines + nice date formatting
      scale_x_date(
        date_breaks = "1 month",
        date_labels = "%b %Y"
      ) +
      
      scale_y_continuous(labels = comma) +
      
      theme_minimal(base_size = 13) +
      theme(
        plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(color = "grey40"),
        
        # Rotate labels for readability
        axis.text.x = element_text(angle = 90, hjust = 1),
        
        # Emphasise monthly gridlines
        panel.grid.major.x = element_line(color = "grey80"),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank()
      )
    
    print(p)
    
    return(list(bp = bp_model, model = model_auto, plot = p))
  }
}

{
  plot_revenue_range <- function(data, start_date, end_date) {
    
    # Ensure date format
    data$Sales.Date <- as.Date(data$Sales.Date)
    start_date <- as.Date(start_date)
    end_date <- as.Date(end_date)
    
    # Filter data
    df <- data[data$Sales.Date >= start_date & data$Sales.Date <= end_date, ]
    
    # Plot
    p <- ggplot(df, aes(x = Sales.Date, y = Total)) +
      geom_line(linewidth = 1, colour = "#2C3E50") +
      
      labs(
        title = paste("Revenue from", start_date, "to", end_date),
        x = "Date",
        y = "Total Revenue"
      ) +
      
      scale_x_date(date_breaks = "1 month", date_labels = "%b %Y") +
      
      theme_minimal(base_size = 13) +
      theme(
        plot.title = element_text(face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid.minor = element_blank()
      )
    
    print(p)
    return(p)
  }
}