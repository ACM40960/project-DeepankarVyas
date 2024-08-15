  #' ---
  #' title: "Dataset Preparation"
  #' author: Deepankar Vyas, 23200527
  #' format:
  #'   html:
  #'     embed-resources: true
  #'   pdf: default
  #' execute: 
  #'   warning: false
  #' ---
  #' 
  #' ## DATASET PREPARATION
  #' 
  #' Here, we will web scrape data from `Football.co.uk[1]` website, for the last 9 seasons of the German league - 
  #' Bundesliga. This dataset will help us provide various match stats for both Home and Away teams such as :-
  #' 
  #' 1.  Goals
  #' 2.  Shots on Target
  #' 3.  Corners
  #' 4.  Fouls Commited
  #' 5.  Yellow and Red Cards
  #' 6.  Betting odds of various organizations
  #' 
  #' We will be using Goals, Shots on Target , Corners of both the Home and Away teams as well as Result of the matches. 
  #' We will scarp data of the past 9-10 seasons i.e (2012/13-2023/24)
  #' 
  #' 
  #' 
  #' Installing packages if not installed
  #' 
  #' 
  #' # Vector of required packages
  
data_prep <- function(){
  packages <- c("knitr","httr","rvest","dplyr","stringr","lubridate","tidyr","furrr", "ggplot2", 
                "data.table","readr", "purrr","rstudioapi","zoo","tidyverse","tidytext","syuzhet",
                "sentimentr","lexicon","textdata","caret","wordcloud","RColorBrewer","plotly","GGally","gridExtra",
                "nortest","ggpubr","reshape2","car","Boruta","randomForest","doParallel","grid","e1071","keras","tfruns",
                "pROC","MASS","ROCR","tensorflow","caret","kableExtra","multiROC","BBmisc","polite")  # Replace with your desired packages
  
  # Function to check if a package is installed, and install it if not
  install_if_missing <- function(packages) {
    for (pkg in packages) {
      if (!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(pkg)
      }
    }
  }
  
  # Install any missing packages
  install_if_missing(packages)
  
  
  ## -----------------------------------------------------------------------------
  #| label: dataset preparation - loading libraries
  #| echo: false
  
  library(knitr)
  library(rvest)
  library(dplyr)
  library(stringr)
  library(tidyr)
  library(furrr)
  library(data.table)
  library(purrr)
  library(rstudioapi)
  library(httr)
  library(polite)
  library(lubridate)
  
  #' 
  ## -----------------------------------------------------------------------------
  #| label : Dataset preparation-football.co.uk
  
  # Setting working directory to Source File location
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  
  # Base URL
  base_url <- "https://www.football-data.co.uk/mmz4281"
  
  # Seasons
  seasons <- c("1213", "1314", "1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324")
  
  # Directory to save files
  dir.create("Bundesliga_2012_2024", showWarnings = FALSE)
  directory <- "Bundesliga_2012_2024"
  
  # Function to download and read a CSV file
  download_and_read_csv <- function(season) {
    file_url <- paste0(base_url, "/", season, "/D1.csv")
    destfile <- paste0(directory, "/D1_", season, ".csv")
    
    # Downloading the file
    download.file(file_url, destfile, mode = "wb")
    
    # Reading the CSV file
    df <- read.csv(destfile)
    
    invisible (df)
  }
  
  # Downloading and combining all CSV files into a master dataset
  master_dataset <- bind_rows(lapply(seasons, download_and_read_csv))
  
  # Converting Date to Date type
  master_dataset$Date <- dmy(master_dataset$Date)
  
  
  
  
  
  #' 
  #' Now, we will web scrap player and team's ratings from `FIFA Index[2]` website staring from FIFA 12 till FIFA 24. 
  #' These are the ratings given by FIFA video game to teams and players and are generated periodically, taking into 
  #' account current form of teams and players, and thus are a good measure of the current state of the team. We will 
  #' first focus on the team ratings. We will web scrap following components of team ratings-
  #' 
  #' 1.  Attack Rating
  #' 2.  Defense Rating
  #' 3.  Midfield Rating
  #' 4.  Overall Rating
  #' 
  ## -----------------------------------------------------------------------------
  #| label: Dataset preparation-fifa_ratings_team
  
  
  
  # Prompt user with a yes/no question
  cat("The web scraping process will take 120+ hours to complete. If No is selected, 
                        data will be taken from the dataset already scraped. Do you want to proceed? (yes/no): ")
  
  
  # Read the user's input
  user_response <- scan(what = character(), nmax = 1, quiet = TRUE)
  
  # Check if the input was provided
  if (length(user_response) == 0) {
    stop("No input provided. Exiting the script.")
  }
  
  # Check user response and proceed accordingly
  if (tolower(user_response) == "yes") {
    # User chose "Yes"
    cat("Proceeding with the long web scraping process...\n")
  
  
  # Base URL components
  base_url <- "https://fifaindex.com/teams/fifa24"
  league_params <- "league=19&league=20&order=desc"
  max_pages <- 3
  
  # Function to scrape data from a given URL
  scrape_fifa_data <- function(session) {
    page <- try(scrape(session))
    
    if (inherits(page, "try-error") || is.null(page)) {
      return(data.frame()) # Return empty data frame if error occurs or no content
    }
    
    # Extracting league names
    leagues <- page %>% html_nodes(".link-league") %>% html_text(trim = TRUE)
    
    # Validating league of the first two teams
    if (length(leagues) < 2 || !all(leagues[1:2] %in% c("Bundesliga", "Germany 1. Bundesliga (1)", "Bundesliga 2", "Germany 2. Bundesliga (2)"))) {
      return(data.frame()) # Return empty data frame if the first two teams' leagues are not Bundesliga
    }
    
    # Extracting team names
    teams <- page %>% html_nodes(".link-team") %>% html_text(trim = TRUE)
    teams <- teams[seq(2, length(teams), 2)] # Skipping the league name links
    
    # Extracting ratings
    att <- page %>% html_nodes("td[data-title='ATT'] span") %>% html_text(trim = TRUE)
    mid <- page %>% html_nodes("td[data-title='MID'] span") %>% html_text(trim = TRUE)
    def <- page %>% html_nodes("td[data-title='DEF'] span") %>% html_text(trim = TRUE)
    ovr <- page %>% html_nodes("td[data-title='OVR'] span") %>% html_text(trim = TRUE)
    
    if (length(att) == 0 || length(mid) == 0 || length(def) == 0 || length(ovr) == 0) {
      return(data.frame()) # Return empty data frame if no ratings found
    }
    
    # Extracting the date
    date <- page %>% html_nodes("a.dropdown-toggle") %>% html_text(trim = TRUE) %>% .[6]
    print(date)
    # Creating the data frame
    data <- data.frame(
      Team = teams,
      Date = date,
      Attack = as.numeric(att),
      Midfield = as.numeric(mid),
      Defense = as.numeric(def),
      Overall = as.numeric(ovr),
      stringsAsFactors = FALSE
    )
    
    return(data)
  }
  
  # Initializing an empty data frame to store the results
  all_fifa_data <- data.frame()
  
  # Bow to the base URL to create a polite session
  session <- bow("https://fifaindex.com")
  
  # Looping through all editions and their possible date values
  for (id in 8:612) {
    for (page in 1:max_pages) {
      # Constructing URL
      print(id)
      if (page == 1) {
        url <- paste0(base_url, "_", id, "/?", league_params)
      } else {
        url <- paste0(base_url, "_", id, "/?page=", page, "&", league_params)
      }
      
      # Navigate to the constructed URL using the session
      session <- nod(session, path = url)
      
      # Scraping data from the URL using the polite session
      data <- scrape_fifa_data(session)
      if (nrow(data) == 0) break # If no data, exit the page loop
      
      # Appending to the master dataset
      all_fifa_data <- bind_rows(all_fifa_data, data)
    }
  }
  
  # Scraping for June 4, 2024
  urls <- c(
    "https://fifaindex.com/teams/?league=19&league=20&order=desc",
    "https://fifaindex.com/teams/?page=2&league=19&league=20&order=desc"
  )
  
  for (url in urls) {
    # Navigate to the constructed URL using the session
    session <- nod(session, path = url)
    
    # Scrape the data
    data <- scrape_fifa_data(session)
    
    if (nrow(data) > 0) {
      all_fifa_data <- bind_rows(all_fifa_data, data)
    }
  }
  
  # Function to standardize month abbreviations
  standardize_month_abbreviations <- function(date_string) {
    date_string <- str_replace_all(date_string, "Sept\\.", "Sep.")
    return(date_string)
  }
  
  all_fifa_data$Date <- sapply(all_fifa_data$Date, standardize_month_abbreviations)
  
  # Converting Date to Date type
  all_fifa_data$Date=mdy(all_fifa_data$Date)
  
  # Ordering by date
  all_fifa_data <- all_fifa_data |> arrange(desc(Date))
  
  
  
  
  #' 
  #' The name of the teams are worded differently in both these websites . Therefore, we are using the names available 
  #' in \``Football.co.uk[1]`\` as the reference, and combining the names to have a standard naming for all the teams.
  #' 
  ## -----------------------------------------------------------------------------
  #| label : Heuristic to combine team names
  
  unique_teams_master <- unique(c(master_dataset$HomeTeam, master_dataset$AwayTeam))
  
  # Defining a mapping of alternative team names to standardized team names in master_dataset
  team_name_mapping <- list(
    "Bayern Munich" = c("Bayern München", "FC Bayern München", "FC Bayern Munich"),
    "Dortmund" = c("Borussia Dortmund"),
    "RB Leipzig" = c("RB Leipzig"),
    "Leverkusen" = c("Bayer 04 Leverkusen", "Leverkusen"),
    "Ein Frankfurt" = c("Eintracht Frankfurt", "Frankfurt"),
    "Hoffenheim" = c("TSG 1899 Hoffenheim", "TSG Hoffenheim", "1899 Hoffenheim"),
    "Wolfsburg" = c("VfL Wolfsburg"),
    "Freiburg" = c("SC Freiburg", "Sport-Club Freiburg"),
    "Union Berlin" = c("1. FC Union Berlin", "Union Berlin"),
    "M'gladbach" = c("Borussia Mönchengladbach", "Borussia M'gladbach", "M'gladbach", "Borussia Mönchengladbach"),
    "Mainz" = c("1. FSV Mainz 05"),
    "Werder Bremen" = c("Werder Bremen", "SV Werder Bremen"),
    "Augsburg" = c("FC Augsburg"),
    "Stuttgart" = c("VfB Stuttgart"),
    "FC Koln" = c("1. FC Köln"),
    "Bochum" = c("VfL Bochum", "VfL Bochum 1848"),
    "Hertha" = c("Hertha BSC", "Hertha Berlin", "Hertha BSC Berlin"),
    "Heidenheim" = c("1. FC Heidenheim", "1. FC Heidenheim 1846", "Heidenheim"),
    "Hamburg" = c("Hamburger SV"),
    "Schalke 04" = c("FC Schalke 04"),
    "Darmstadt" = c("SV Darmstadt 98"),
    "Hannover" = c("Hannover 96"),
    "Fortuna Dusseldorf" = c("Fortuna Düsseldorf", "Düsseldorf"),
    "1. FC Kaiserslautern" = c("1. FC Kaiserslautern", "Kaiserslautern"),
    "Paderborn" = c("SC Paderborn 07"),
    "Karlsruher SC" = c("Karlsruher SC"),
    "FC St. Pauli" = c("FC St. Pauli"),
    "Nurnberg" = c("1. FC Nürnberg", "1. FC Nuremberg"),
    "Hansa Rostock" = c("Hansa Rostock"),
    "1. FC Magdeburg" = c("1. FC Magdeburg"),
    "Holstein Kiel" = c("Holstein Kiel"),
    "Greuther Furth" = c("SpVgg Greuther Fürth", "Fürth"),
    "Braunschweig" = c("Eintracht Braunschweig", "Braunschweig"),
    "VfL Osnabrück" = c("VfL Osnabrück"),
    "SV Elversberg" = c("SV Elversberg"),
    "SV Wehen Wiesbaden" = c("SV Wehen Wiesbaden"),
    "Bielefeld" = c("Arminia Bielefeld", "DSC Arminia Bielefeld"),
    "SV Sandhausen" = c("SV Sandhausen", "SV Sandhausen 1916"),
    "Jahn Regensburg" = c("Jahn Regensburg", "SSV Jahn Regensburg"),
    "Würzburger Kickers" = c("Würzburger Kickers", "FC Würzburger Kickers"),
    "Dynamo Dresden" = c("Dynamo Dresden", "SG Dynamo Dresden"),
    "FSV Frankfurt" = c("FSV Frankfurt", "FSV Frankfurt 1899"),
    "1860 München" = c("1860 München", "TSV 1860 München", "1860 Munich"),
    "FC Energie Cottbus" = c("FC Energie Cottbus", "Energie Cottbus"),
    "Alemannia Aachen" = c("Alemannia Aachen"),
    "Ingolstadt" = c("FC Ingolstadt 04")
  )
  
  # Reversing the mapping for quick lookup
  reverse_mapping <- setNames(rep(names(team_name_mapping), lengths(team_name_mapping)), unlist(team_name_mapping))
  
  # Standardizing team names in all_fifa_data
  all_fifa_data$Team <- unname(reverse_mapping[all_fifa_data$Team])
  
  # Filtering records in all_fifa_data to keep only those teams that are present in master_dataset
  all_fifa_data <- all_fifa_data %>% filter(Team %in% unique_teams_master)
  
  #' 
  #' Now, we will Aggregate Ratings by Month and Year, so that each team has an Attack, Midfield, Defense and Overall 
  #' Rating for each month of the years (2012-2024)
  #' 
  ## -----------------------------------------------------------------------------
  #| label: Dataset-preparation-Aggregate_Ratings_by_Month_and_Year
  
  
  # Extracting Year and Month from Date
  all_fifa_data <- all_fifa_data %>%
    mutate(Year = year(Date), Month = month(Date))
  
  # Calculating average ratings by Team, Year, and Month
  avg_ratings <- all_fifa_data %>%
    group_by(Team, Year, Month) %>%
    summarise(
      Avg_Attack = round(mean(Attack, na.rm = TRUE)),
      Avg_Midfield = round(mean(Midfield, na.rm = TRUE)),
      Avg_Defense = round(mean(Defense, na.rm = TRUE)),
      Avg_Overall = round(mean(Overall, na.rm = TRUE))
    ) %>%
    ungroup()
  
  
  
  
  #' 
  #' Now, we are merging the average ratings with the master dataset. We first found the average ratings of each team 
  #' for each month of the year, wherever applicable and ten merged with the master dataset. We now have a dataset 
  #' where we have ratings of each team (for each month) for each match.
  #' 
  ## -----------------------------------------------------------------------------
  #| label: Dataset-preparation-ratings_merge
  
  
  # Ensuring avg_ratings has continuous month-year data for each team
  all_months <- expand.grid(Year = unique(avg_ratings$Year), Month = 1:12)
  all_teams_months <- expand.grid(Team = unique(avg_ratings$Team), Year = unique(avg_ratings$Year), Month = 1:12)
  
  avg_ratings_full <- all_teams_months %>%
    left_join(avg_ratings, by = c("Team", "Year", "Month")) %>%
    arrange(Team, Year, Month)
  
  # Filling in missing values with the nearest previous available values
  avg_ratings_full <- avg_ratings_full %>%
    group_by(Team) %>%
    fill(Avg_Attack, Avg_Midfield, Avg_Defense, Avg_Overall, .direction = "down") %>%
    ungroup()
  
  # Removing NA values 
  avg_ratings_full <- avg_ratings_full %>%
    filter(!(Year == 2011 & Month < 9))
  
  avg_ratings_full <- avg_ratings_full %>%
    group_by(Team) %>%
    fill(Avg_Attack, Avg_Midfield, Avg_Defense, Avg_Overall, .direction = "up") %>%
    ungroup()
  
  
  
  # Extracting Year and Month from Date in master_dataset
  master_dataset <- master_dataset %>%
    mutate(Year = year(Date), Month = month(Date))
  
  # Joining avg_ratings_full with master_dataset for HomeTeam
  master_dataset <- master_dataset %>%
    left_join(avg_ratings_full, by = c("HomeTeam" = "Team", "Year", "Month")) %>%
    rename(
      Home_Attack = Avg_Attack,
      Home_Midfield = Avg_Midfield,
      Home_Defense = Avg_Defense,
      Home_Overall = Avg_Overall
    )
  
  # Joining avg_ratings_full with master_dataset for AwayTeam
  master_dataset <- master_dataset %>%
    left_join(avg_ratings_full, by = c("AwayTeam" = "Team", "Year", "Month"))  %>% 
    rename(
      Away_Attack = Avg_Attack,
      Away_Midfield = Avg_Midfield,
      Away_Defense = Avg_Defense,
      Away_Overall = Avg_Overall
    )
  
  # Removing extra columns
  master_dataset <- master_dataset %>%
    select(-Year, -Month)
  
  
  
  #' 
  #' Now, we will web scrap pre match expert reports from `whoscored.com`\[3\]. 
  #' This will give us experts' opinion and 
  #' will help us understand whether including experts' opinion help complement predictions generated through 
  #' statistical features alone.
  #' 
  ## -----------------------------------------------------------------------------
  #| label: Dataset-preparation-Pre_match_reports
  
  # Function to scrape match data
  scrape_match_data <- function(match_id) {
    url <- paste0("https://www.whoscored.com/Matches/", match_id, "/Preview/")
    page <- read_html(url)
    
    # Checking if the match is a Bundesliga match
    league <- (page %>% html_nodes("a[href*='Germany-Bundesliga']") %>%
    html_text(trim = TRUE))[1]
  
    league <- str_trim(strsplit(league,split = "-")[[1]][1])
  
    if ((!league == "Bundesliga") || (is.na(league))) {
      return(NULL) # Returning NULL if it is not a Bundesliga match
    }
    
    cat("Match id processing - ", match_id, "\n")
    
    # Scraping relevant content
    script_content <- page %>% html_nodes(xpath = "//script[contains(., 'matchheader')]") %>% html_text()
    date_pattern <- "([0-9]{2}/[0-9]{2}/[0-9]{4})"
    date <- str_extract(script_content, date_pattern)
    
    if (is.na(date)) {
      message("No date found for match ID: ", match_id)
      return(NULL)
    }
    
    match_date <- dmy(date) # Converting to date format
  
    # Extracting Team News and Predictions section
    team_news_elements <- page %>% html_nodes(xpath = '//*[contains(text(),"Team News")]/following-sibling::div') %>% 
      html_text(trim = TRUE)
    prediction <- page %>% html_nodes(xpath = '//*[contains(text(),"Prediction")]/following-sibling::div') %>% 
      html_text(trim = TRUE)
    
    if (length(team_news_elements) == 0) {
      message("No team news found for match ID: ", match_id)
      return(NULL)
    }
  
    # Extracting team names and their respective news
    team_names <- str_split(team_news_elements, "\r\n\\s*\r\n")[[1]] |> str_trim()
    team_news <- str_split(team_news_elements, "\r\n\\s*\r\n")[[2]] |> str_trim()
    
    # Formatting predictions
    prediction_formatted <- str_split(prediction, "\r\n\\s*\r\n")[[1]][1] |> str_trim()
    
    if (length(team_news) < 2) {
      message("Unexpected team news format for match ID: ", match_id)
      return(NULL)
    }
    
    # Formating the team news
    team_news_formatted <- paste0(
      team_names[1], "'s team news states that ", team_news[1],
      team_names[2], "'s team news states that ", team_news[2]
    )
    
    preview <- paste(team_news_formatted, prediction_formatted, sep = "\n\n")
    
    print(match_id)
    
    data.frame(
      Home_Team = team_names[1],
      Away_Team = team_names[2],
      Match_Id = match_id,
      Match_date = match_date,
      Preview = preview,
      stringsAsFactors = FALSE
    )
  }
  
  # Initializing parallel processing
  plan(multisession, workers = 8)  # Using 8 workers
  
  # Range of match IDs
  match_ids <- 621122:1743696
  
  # Spliting the range of match IDs into 8 parts
  split_match_ids <- split(match_ids, cut(match_ids, breaks = 8, labels = FALSE))
  
  # Function to safely scrape and return data
  safe_scrape <- function(match_id) {
    tryCatch({
      scrape_match_data(match_id)
    }, error = function(e) {
      message("Error with match ID: ", match_id, " - ", e$message)
      return(NULL)
    })
  }
  
  # Using furrr::future_map to scrape data in parallel for each part
  all_match_data <- map_dfr(split_match_ids, function(ids) {
    
    match_data_list <- future_map(ids, safe_scrape, .progress = TRUE)
    # Filtering out NULL results and bind rows
    bind_rows(match_data_list)
  })
  
  
  
  
  #' 
  #' Standardizing the team names and merging the resultant dataset with our master dataset.
  #'  Our final dataset would now contain a pre match report for every match played.
  #' 
  ## -----------------------------------------------------------------------------
  #| label: Dataset-preparation-Final_merge
  
  # Mapping of team names
  team_name_mapping <- list(
    "Borussia Dortmund" = "Dortmund",
    "Hannover" = "Hannover",
    "Stuttgart" = "Stuttgart",
    "Freiburg" = "Freiburg",
    "Augsburg" = "Augsburg",
    "Hamburg" = "Hamburg",
    "Eintracht Frankfurt" = "Ein Frankfurt",
    "Greuther Fuerth" = "Greuther Furth",
    "Borussia M.Gladbach" = "M'gladbach",
    "Bayern" = "Bayern Munich",
    "Schalke" = "Schalke 04",
    "Mainz" = "Mainz",
    "Werder Bremen" = "Werder Bremen",
    "Fortuna Duesseldorf" = "Fortuna Dusseldorf",
    "Nuernberg" = "Nurnberg",
    "Hoffenheim" = "Hoffenheim",
    "Leverkusen" = "Leverkusen",
    "Wolfsburg" = "Wolfsburg",
    "Eintracht Braunschweig" = "Braunschweig",
    "Hertha Berlin" = "Hertha",
    "FC Koln" = "FC Koln",
    "Paderborn" = "Paderborn",
    "Darmstadt" = "Darmstadt",
    "Ingolstadt" = "Ingolstadt",
    "RBL" = "RB Leipzig",
    "Union Berlin" = "Union Berlin",
    "Arminia Bielefeld" = "Bielefeld",
    "Bochum" = "Bochum",
    "FC Heidenheim" = "Heidenheim"
  )
  
  # Standardizing team names in all_match_data
  all_match_data <- all_match_data %>%
    mutate(
      Home_Team = unlist(lapply(Home_Team, function(x) team_name_mapping[[x]])),
      Away_Team = unlist(lapply(Away_Team, function(x) team_name_mapping[[x]]))
    )
  
  # Ordering both datasets by date
  all_match_data <- all_match_data %>%
    arrange(Match_date)
  
  master_dataset <- master_dataset %>%
    arrange(date)
  
  # Appending preview column to master_dataset
  master_dataset <- master_dataset %>%
    left_join(all_match_data %>% select(Home_Team, Away_Team, Match_date, Preview),
              by = c("HomeTeam" = "Home_Team", "AwayTeam" = "Away_Team", "Date" = "Match_date"))
  
  
  master_dataset <- master_dataset[!is.na(master_dataset$Preview) 
                                   & master_dataset$Preview != "", ]
  
  
  } else {
    # User chose "No"
    cat("Skipping the web scraping process...\n")
    
    km <- read.csv("../Dataset/Master_Dataset_Before_Preprocessing.csv")
    
    km$Date <- as.Date(km$Date, format = "%Y-%m-%d")
    
    
    master_dataset <- master_dataset %>%
      left_join(km %>% select(Date, HomeTeam, AwayTeam,
                              Home_Attack, Home_Midfield, Home_Defense, Home_Overall,
                              Away_Attack, Away_Midfield, Away_Defense, Away_Overall,Preview),
                by = c("Date", "HomeTeam", "AwayTeam"))
    
    master_dataset <- master_dataset %>%
      filter(!is.na(Home_Attack))
  
  }
  
  #' 
  #' Adding `Season` and `Match_Time` columns:-
  #' 
  ## -----------------------------------------------------------------------------
  #| label: Dataset-preparation-Season_and_Match_Time
  
  # Function to determine the season based on the month
  get_season <- function(date) {
    month <- month(date)
    if (month %in% c(3, 4, 5)) {
      return("Spring")
    } else if (month %in% c(6, 7, 8)) {
      return("Summer")
    } else if (month %in% c(9, 10, 11)) {
      return("Autumn")
    } else if (month %in% c(12, 1, 2)) {
      return("Winter")
    } else {
      return(NA)
    }
  }
  
  # Adding Season column to the dataset
  master_dataset$Season <- sapply(master_dataset$Date, get_season)
  
  
  # Saving the updated master_dataset to a higher-level directory
  output_path <- "../Dataset/Master_Dataset.csv"
  
  # Adding Day_of_Week column to the dataset
  master_dataset$Day_of_Week <- wday(master_dataset$Date, label = TRUE)
  
  # Saving the master_dataset to the specified path
  write.csv(master_dataset, file = output_path, row.names = FALSE)
  
  return(master_dataset)
}

master_dataset <- data_prep()


