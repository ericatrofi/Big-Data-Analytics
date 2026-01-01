library(arrow)
library(dplyr)
library(pryr)
library(lubridate)
library(ggplot2)
library(tidyverse)


mem_used()

system.time({
  ds <- open_dataset(
    "gs://bigdata_project_dataset/ecommerce/2019-Nov.csv.parquet",
    format = "parquet"
  )
})

mem_used() # we can see it did not increase

system.time({
  mem_change(
    purchases <- ds |>
      select(user_session, user_id, event_type) |>
      filter(event_type == "purchase") |>
      distinct(user_session) |>
      mutate(n_purchase = 1L)
  )
})


system.time({
  mem_change(
    session_features <- ds |> 
      select(user_session, user_id, event_type, price, event_time, category_id) |>       
      group_by(user_session, user_id) |>
      summarise(
        n_view = sum(event_type == "view", na.rm=TRUE),
        n_cart = sum(event_type == "cart", na.rm=TRUE),
        avg_price = mean(price, na.rm = TRUE),
        session_start = min(event_time, na.rm = TRUE),
        session_end = max(event_time, na.rm = TRUE),
        n_unique_categories = n_distinct(category_id),
        
        .groups = "drop") |>
      left_join(purchases, by = c("user_session")) |>
      mutate(n_purchase = coalesce(n_purchase, 0L)) |>
      select(
        -user_session, -user_id
      ) |>
      collect()
  )
})


system.time({
  mem_change(
    session_features <- session_features |>
      mutate(
        session_start = ymd_hms(session_start, tz = "UTC"),
        session_end = ymd_hms(session_end, tz = "UTC"),
        session_duration = as.numeric(difftime(session_end, session_start, units = "mins")),
        is_weekend = as.integer(wday(session_start, week_start = 1) >= 6),
        is_beginning_of_month = as.integer(day(session_start) <= 3)
      ) |>
      select(
        -session_start, 
        -session_end
      )
  )
})

mem_used()

object_size(session_features)


# VISUAL OF CLEANED DATASET


#N Counts



counts <- session_features %>%
  summarise(across(c(n_view, n_cart, n_purchase), ~sum(., na.rm = TRUE))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "count")

ggplot(counts, aes(x = column, y = count)) +
  geom_col(fill = "steelblue") +
  labs(title = "Sum of Values per Column", y = "Sum", x = "Column") +
  theme_minimal()
ggsave("Clean_count.png", plot = last_plot(), width = 6, height = 4)

ggplot(counts, aes(x = "", y = count, fill = column)) +
  geom_col(width = 1) +
  coord_polar("y", start = 0) +
  labs(title = "Share of Total Activity", fill = "Metric") +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5))

# Unique catagories
summary(session_features$n_unique_categories)


## Avg price
ggplot(session_features, aes(x = avg_price)) +
  geom_histogram(bins = 80, color = "white") +
  theme_bw()
ggsave('Clean_2_hist.png', plot = last_plot(), width = 6, height = 4)

summary(session_features$avg_price)

## Session Duration
summary(session_features$session_duration)

ggplot(session_features, aes(x = session_duration + 1)) +
  geom_histogram(bins = 100, color = "white") +
  scale_x_log10() +
  coord_cartesian(xlim = c(1, 150)) +  # zoom in to 1-150
  labs(x = "session_duration + 1 (log10 scale)") +
  theme_bw()

ggsave('Clean_3_duration.png', plot = last_plot(), width = 6, height = 4)


