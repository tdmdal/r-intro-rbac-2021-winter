# load the libraries
library(readr)
library(dplyr)
library(huxtable)

# specify the dataset url
data_url <- "https://tdmdal.github.io/r-intro-rbac-2021-winter/data/hprice.csv"

# read it into a dataframe, hprice; use the read_csv() function in readr library
hprice <- read_csv(data_url)

# print out the first few rows of the data
head(hprice)

# summary statistics
summary(hprice)

# data manipulation using dplyr package
# filter out rows with prices > 95 percentile or < 5 percentile
# an alternative way: hprice_reg <- filter(hprice, price < quantile(hprice$price, 0.95) & price > quantile(hprice$price, 0.05))
hprice_reg <- filter(hprice, percent_rank(price) < 0.95 & percent_rank(price) > 0.05)

# create new variables lprice, lnox, ldist; mutate()
hprice_reg <- mutate(hprice_reg, lprice = log(price), lnox = log(nox), ldist = log(dist))

# not necessary for our regression, but for practice, let's select only lprice, lnox, ldist, rooms, stratio columns; select()
hprice_reg <- select(hprice_reg, lprice, lnox, ldist, rooms, stratio)

# can we combine the above three operations? pipe %>%
hprice_reg <- hprice %>%
  filter(percent_rank(price) < 0.95 & percent_rank(price) > 0.05) %>%
  mutate(lprice = log(price), lnox = log(nox), ldist = log(dist)) %>%
  select(lprice, lnox, ldist, rooms, stratio)

# three regression models
# 1. simple regression; lm()
lr_simple <- lm(formula = lprice ~ lnox, data = hprice_reg)

# 2. multiple regression; lm()
lr_multiple1 <- lm(formula = lprice ~ lnox + rooms, data = hprice_reg)

# 3. multiple regression with interactive terms; lm()
lr_multiple2 <- lm(lprice ~ lnox + ldist + rooms + I(rooms^2) + stratio, data = hprice_reg)

# report regression result; summary()
summary(lr_simple)
summary(lr_multiple1)
summary(lr_multiple2)

# can we do better; huxtable package
huxreg(lr_simple, lr_multiple1, lr_multiple2)

coeffs_text <- c("(Intercept)", 
                 "log nox" = "lnox",
                 "rooms",
                 "log distance" = "ldist",
                 "rooms * rooms" = "I(rooms^2)",
                 "student-teacher ratio" = "stratio")
reg_result <- huxreg(lr_simple, lr_multiple1, lr_multiple2, coefs = coeffs_text)
reg_result %>%
  set_caption("<b>Regression Results</b>") %>%
  theme_article() %>%
  quick_html(file = "result.html")

