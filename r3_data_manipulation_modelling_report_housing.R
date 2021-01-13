# load the libraries


# specify the dataset url
data_url <- "https://tdmdal.github.io/r-intro-rbac-2021-winter/data/hprice.csv"

# read it into a dataframe, hprice; use the read_csv() function in readr library


# print out the first few rows of the data


# summary statistics


# data manipulation using dplyr package
# filter out rows with prices > 95 percentile or < 5 percentile


# create new variables lprice, lnox, ldist; mutate()


# not necessary for our regression, but for practice, let's select only lprice, lnox, ldist, rooms, stratio columns; select()


# can we combine the above three operations? pipe %>%


# three regression models
# 1. simple regression (lnox); lm()


# 2. multiple regression (lnox, rooms); lm()


# 3. multiple regression with interactive terms (lnox, ldist, rooms, rooms^2, stratio); lm()


# report regression result; summary()


# can we do better; huxtable package

