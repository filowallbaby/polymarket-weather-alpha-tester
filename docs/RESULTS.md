# Results

Development stopped after the October 2026 review.

The post-registration replay had 19 signals and 3 wins, against a mean predicted probability of 54.1%. One share per signal returned $0.073 total after observed asks, modeled fees and a $0.02 buffer. One additional cent of cost per signal made the result negative.

Signals were reconstructed days later. These were not verified fills, and the registered experiment never met its coverage requirements.

Known issues:

- 17 of 39 model features were missing from the reviewed live feature files.
- Historical US temperature changes used Fahrenheit; pretraining used Celsius.
- Training covered 09:00, 12:00 and 15:00; live scoring covered the full day.
- Quote validation did not consistently reject invalid or stale books.

No tradable edge established. The implementation remains uncorrected.
