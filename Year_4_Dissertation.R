# Goals 

# Spatial distribution (with 2001 data)
# RQ: Urban vs Rural (C-PCA)
# RQ: Recreational vs Industrial vs Residential vs Crop (C-PCA)
# From here, identify element of interest for temporal analysis 
# Conclusion: In the past, X elements are elevated in urban areas, 
# and X elements mostly affected these landuses.
# Leading question: Now that industrial development has waned over time, have these elements degraded? 
# have they accumulated? how does top soil retain heavy metal pollution?

#Temporal distribution (with 2001-2010-2018 data)
# RQ: specific element change from 2001-2010-2018 of heavy metals south-east of glasgow city center (box-plots, ANOVA)

#Causes
# RQ: what is the source of this specific element change? 
# River Clyde source: Investigate the how X element change with distance from river
# Road source: Investigate the how X element change with distance from road
# Mine source: Investigate the how X element change with distance from mines

#-----------------------------------------------------------------------------------------------------------------

#Load packages

library(readxl)
library(magrittr)
library(tidyverse)
library(GeochemUtils)
library(robCompositions)
library(StatDA)
library(egg)
library(ggrepel)
library(sf)
library(compositions)
library(data.table)
library(ggplot2)
library(dplyr)
library(rgdal)
library(igraph)
library(compositions)
library(stringr)
library(openxlsx)

source("/Users/Jasthecoolbean/Desktop/PCA_barium/Fct_ggplot-Extensions.R")
source("/Users/Jasthecoolbean/Desktop/PCA_barium/Fct_Subfunctions.R")
source("/Users/Jasthecoolbean/Desktop/PCA_barium/Fct_HelperFunction.R")

# Data cleaning
  
    # Make a combined data file with additional columns specifying whether data point is
    # A) Type: rural or urban [DONE]
    # B) Year: 2001, 2002, 2010, 2011 or 2018 [DONE]
    # C) from 2018 [DONE]
    # D) Landuse: recreational, industrial, residential or crop [DONE]

    # Create functions to clean both top soil and deeper soil of weird symbols 

    # Then, normalize these topsoil values against deeper soil values to get human-added heavy metals
    # E) Make a new data table, that will compute top soil - deep soil for each data point. 
      # Retain all other information, such as columns with landuse, year, type

#--------------------------------------------------------------------------------------------------------#
  
### A: Rural or Urban

# Classification of urban areas are given, but I do not have the classification for rural area. 
# I will figure this out by subtracting rural soil from total soil

urbansoil=read_csv("/Users/Jasthecoolbean/Desktop/Year 4/dissertation/Urban_Only/Urban_soils.csv")%>%data.table()
allsoil=read.xlsx("/Users/Jasthecoolbean/Desktop/Year 4/dissertation/XYGBASE_Shallow_Soils.xlsx")%>%data.table()

# I can find the rural soil by finding coordinates present in all soil that is not present in urbansoil.

ruralsoil=anti_join(allsoil,urbansoil, by = c("X_COORD", "Y_COORD"))

# NOTE: rural soil + urbansoil =/= allsoil, because in the urban soil data, there is an additional sampling in the westzone
# Hence, after isolating rural soil, I should combine with urbansoil to get a new, updated allsoil dataset.

#Add new column 'Type' in both data table, and making all other column names match

urbansoil$Type <- 'Urban'
ruralsoil$Type <- 'Rural'
colnames(urbansoil)[colnames(urbansoil) == "URBAN_CENT"] <- "ATLAS_CODE"

colnames(urbansoil)

# Make an updated allsoil dataset
allsoil<- rbind(urbansoil, ruralsoil)

#--------------------------------------------------------------------------------------------------------#

### B: Year

# Change full date into Year. First I'll change the column name.
colnames(allsoil)[colnames(allsoil)=="DATE_VISIT"]<-"Year"

# Then, I convert this column into a recognizable year-month-date format
allsoil$Year <- as.Date(allsoil$Year, format = "%d/%m/%Y")

# Extract the year from the column
allsoil$Year <- format(allsoil$Year, "%Y")

#--------------------------------------------------------------------------------------------------------#

### C: 2018 Clean and combine

#Load 2018 data + tidy to match allsoil

soil2018 <- read_csv("/Users/Jasthecoolbean/Desktop/Year 4/dissertation/2018_soil.csv")%>%data.table()
colnames(soil2018)
soil2018 <- soil2018[-c(1:4),-c(1:2,5:6,62:172)]
colnames(soil2018)[colnames(soil2018)=="Easting"]<-"X_COORD"
colnames(soil2018)[colnames(soil2018)=="Northing"]<-"Y_COORD"
allsoil <- allsoil[,-c(1:5,7)]
soil2018$Year<-2018
soil2018$Type<-"Urban"

#I need to tidy the column names for elements in both allsoil and soil2018

colnames(allsoil) <- gsub("_XRF|_AAS", "", colnames(allsoil))

allsoil<-allsoil[,-c("LOI","pH")]
                 
allsoil<-allsoil[,-23]

colnames(allsoil)

soil2018<- soil2018[,-c("pH")]

#Abbreviating element names in soil2018 

element_mapping <- c(
  "Aluminium" = "Al", "Calcium" = "Ca", "Iron" = "Fe", "Potassium" = "K",
  "Magnesium" = "Mg", "Manganese" = "Mn", "Sodium" = "Na", "Phosphorus" = "P",
  "Silicon" = "Si", "Titanium" = "Ti", "Silver" = "Ag", "Arsenic" = "As",
  "Barium" = "Ba", "Bismuth" = "Bi", "Bromine" = "Br", "Cadmium" = "Cd",
  "Cerium" = "Ce", "Chlorine" = "Cl", "Cobalt" = "Co", "Chromium" = "Cr",
  "Caesium" = "Cs", "Copper" = "Cu", "Gallium" = "Ga", "Germanium" = "Ge",
  "Hafnium" = "Hf", "Mercury" = "Hg", "Iodine" = "I", "Indium" = "In",
  "Lanthanum" = "La", "Molybdenum" = "Mo", "Niobium" = "Nb", "Neodymium" = "Nd",
  "Nickel" = "Ni", "Lead" = "Pb", "Rubidium" = "Rb", "Sulphur" = "S",
  "Antimony" = "Sb", "Scandium" = "Sc", "Selenium" = "Se", "Samarium" = "Sm",
  "Tin" = "Sn", "Strontium" = "Sr", "Tantalum" = "Ta", "Tellurium" = "Te",
  "Thorium" = "Th", "Thallium" = "Tl", "Uranium" = "U", "Vanadium" = "V",
  "Tungsten" = "W", "Yttrium" = "Y", "Ytterbium" = "Yb", "Zinc" = "Zn",
  "Zirconium" = "Zr", "Chromium" = "Cr" , "ChromiumVI" = "CrVI"
)

colnames(soil2018) <- element_mapping[colnames(soil2018)]
soil2018<-soil2018[,-c("CrVI")]

#Combining the 2001,2010 and 2018 data 
allsoil<-rbind(allsoil,soil2018)
allsoil<-allsoil[-c(3348:3399),]

#--------------------------------------------------------------------------------------------------------#

### D: Landuse

# I clipped the landuses from Verisk Digimap to the all soil data file. 
# I will load the soil data found in each landuse

indus<-read.csv("/Users/Jasthecoolbean/Desktop/Year 4/dissertation/landuse/indus_soil.csv")%>%as.data.table()
recrea<-read.csv("/Users/Jasthecoolbean/Desktop/Year 4/dissertation/landuse/recrea_soil.csv")%>%as.data.table()
res<-read.csv("/Users/Jasthecoolbean/Desktop/Year 4/dissertation/landuse/res_tot_soil.csv")%>%as.data.table()
agri<-read.csv("/Users/Jasthecoolbean/Desktop/Year 4/dissertation/landuse/agri_soil.csv")%>%as.data.table()

# new column in files called landuse
allsoil$landuse<-NA
indus$landuse<-"industrial"
agri$landuse<-"agricultural"
recrea$landuse<-"recreational"
res$landuse<-"residential"

files<-c(indus,agri,recrea,res)

#Matching X, Y COORD and replacing the entire row with the landuse files

library(data.table)

# Assuming allsoil, indus, agri, res, and recrea are data.tables
allsoil <- data.table(allsoil)
indus <- data.table(indus)
agri <- data.table(agri)
res <- data.table(res)
recrea <- data.table(recrea)
allsoil$landuse<-as.character(allsoil$landuse)

tail(allsoil,10)

# Set keys for all data.tables
setkey(allsoil, X_COORD, Y_COORD)
setkey(indus, X_COORD, Y_COORD)
setkey(agri, X_COORD, Y_COORD)
setkey(res, X_COORD, Y_COORD)
setkey(recrea, X_COORD, Y_COORD)

# Create a temporary column to identify rows with a match
allsoil[, match := FALSE]

# Update rows only when coordinates match and set match to TRUE
allsoil[indus, `:=`(landuse = i.landuse, match = TRUE), on = .(X_COORD, Y_COORD)]
allsoil[agri, `:=`(landuse = i.landuse, match = TRUE), on = .(X_COORD, Y_COORD)]
allsoil[res, `:=`(landuse = i.landuse, match = TRUE), on = .(X_COORD, Y_COORD)]
allsoil[recrea, `:=`(landuse = i.landuse, match = TRUE), on = .(X_COORD, Y_COORD)]

# Check the number of occurrences of 'agricultural' in landuse
landuse_count <- sum(grepl('industrial', allsoil$landuse, ignore.case = TRUE))
cat("Number of occurrences of 'agricultural' in landuse:", landuse_count, "\n")

#reordering the columns
setcolorder(allsoil, c(names(allsoil)[(ncol(allsoil)-1):ncol(allsoil)], setdiff(names(allsoil), names(allsoil)[(ncol(allsoil)-1):ncol(allsoil)])))
allsoil<-allsoil[,-2]

#Saving the new combined dataset as allsoil

directory_path <- "/Users/Jasthecoolbean/Desktop/Year 4/dissertation"
file_path <- file.path(directory_path, "allsoil.csv")
write.csv(allsoil, file_path, row.names = FALSE)

#--------------------------------------------------------------------------------------------------------#

### E: Deeper soil and allsoil clean symbols

#Load deeper soil file

allsoil<-read_csv("/Users/Jasthecoolbean/Desktop/Year 4/dissertation/allsoil.csv")
deepsoil<-read_csv("/Users/Jasthecoolbean/Desktop/Year 4/dissertation/Deeper_Soil/Deeper_soil.csv")

#Clean up column names

to_remove<-c("Ag_DCOES","Al_DCOES","Ba_DCOES","Bi_DCOES","Ca_DCOES","Cd_DCOES","Ce_DCOES","Co_DCOES","Cr_DCOES","Cu_DCOES","Fe_DCOES","Ga_DCOES","K_DCOES","La_DCOES","LOI450_LOI", "Mg_DCOES","Mn_DCOES","Mo_DCOES","Nb_DCOES","Ni_DCOES","P_DCOES","Pb_DCOES","pH","Rb_DCOES","Sc_DCOES","Si_DCOES","Sn_DCOES","Sr_DCOES","Ti_DCOES","V_DCOES","Y_DCOES","Zn_DCOES","Zr_DCOES")
deepsoil<-deepsoil[,setdiff(colnames(deepsoil),to_remove)]
colnames(deepsoil) <- gsub("_XRF|_DCOES|_DNA", "", colnames(deepsoil))
deepsoil<-deepsoil[,-c(1:3,5)]
colnames(deepsoil)[colnames(deepsoil) == "DATE_VISIT"] <- "Year"
deepsoil$Year <- as.Date(deepsoil$Year, format = "%d/%m/%Y")
deepsoil$Year <- format(deepsoil$Year, "%Y")

#Clean NA

# I cannot just do na.omit because most columns contain at least one NA
# Before I apply na.omit, I will try to remove main columns that contain lots of NA

# Function to remove entire column of elements if there is more than a certain amount of NA
  # w is data table, x and y is range of columns, z is number of NA

NA_omit <- function(w, x, y, z) {
  allele <- w[, x:y]
  exceedthreshold <- names(allele)[colSums(is.na(allele)) > z]
  print(exceedthreshold)
  w <- w[, !(names(w) %in% exceedthreshold)]
  return(w)
}

#Apply to allsoil and deepsoil (35 chosen as it does not remove main harmful pollutant like lead, copper etc)

allsoil <- NA_omit(allsoil, 5, 57, 35)
deepsoil <- NA_omit(deepsoil, 4, 54, 35)

#Apply na.omit

allsoil<-na.omit(allsoil)
deepsoil<-na.omit(deepsoil)

# Create a function to clean symbols, and output them as -detection value

# Symbols and what I will do with them:

# '$' change number behind $ to the detection value of element, then impute
# '#' multiply the value by 2, then impute
# '>,<,&' remove symbols
# '!,?,~,\' make NA 

# Function to clean '$'

elements<-c("Ag","Al","As","Ba",
            "Bi","Br","Ca","Cd","Ce","Cl","Co","Cr","Cs",
            "Cu","Fe","Ga","Ge","Hf","Hg","I","In","K","La",
            "Mg","Mn","Mo","Na","Nb","Nd","Ni","P","Pb","Rb",
            "S","Sb","Sc","Se","Si","Sm","Sn","Sr","Ta",
            "Te","Th","Ti","Tl","U","V","W","Y","Yb","Zn","Zr")
            
            
#when detection limit is N/A

detect_limit<-c(-0.5,-1000,-0.9,-1,
                -0.3,-0.8,-350,-0.5,-1,-200,-1.5,-3,-1,
                -1.3,-70,-1,-0.5,-1,-0.5,-0.5,-0.5,-80,-1,
                -1800,-40,-0.2,-2200,-1,-4,-1.3,-220,-1.3,-1,
                -1000,-0.5,-3,-0.2,-450,-3,-0.5,-1,-1,
                -0.5,-0.7,-60,-0.5,-0.5,-3,-0.6,-1,-1.5,-1.3,-1)

omit_dollar <- function(df, elements, detect_limit) {
  # Replace cells containing '$' with column names
  for (col in names(df)) {
    if (any(grepl("\\$", df[[col]]))) {
      df[[col]][grepl("\\$", df[[col]])] <- col
    }
  }
  
  # Replace specific element names with corresponding detection limits for each existing column
  for (i in seq_along(elements)) {
    element <- elements[i]
    limit <- detect_limit[i]
    
    if (element %in% colnames(df)) {
      df[[element]] <- lapply(df[[element]], function(x) ifelse(grepl(element, x), as.character(limit), x))
    }
  }
  
  # Remove all $ symbols from the entire data frame
  df[] <- lapply(df, function(x) gsub("\\$", "", x))
  
  return(df)
}

allsoil <- omit_dollar(allsoil, elements, detect_limit)

#Function to clean '#'

# Iterate over each column in the data frame

omit_hash <- function(df) {
  for (col in colnames(df)) {
    # Iterate over each element in the column
    for (i in 1:length(df[[col]])) {
      # Check if the element starts with "#" and remove it
      if (grepl("^#", df[[col]][i])) {
        # Extract the numeric part and convert it to a number
        value <- as.numeric(gsub("^#", "", df[[col]][i]))
        # Double the value and add a negative sign
        df[[col]][i] <- paste0("-", value * 2)
      }
    }
  }
  return(df)
}

deepsoil <- omit_hash(deepsoil)
allsoil <- omit_hash(allsoil)

# Function to clean '>,<,&'

omit_symb <- function(df) {
  for (col in colnames(df)) {
    # Iterate over each element in the column
    for (i in 1:length(df[[col]])) {
      # Remove symbols ><&
      df[[col]][i] <- gsub("[><&]", "", df[[col]][i])
    }
  }
  return(df)
}

allsoil <- omit_symb(allsoil)
deepsoil <- omit_symb(deepsoil)

# Function to clean '!,?,~,\' make NA

NA_symb <- function(df) {
  for (col in colnames(df)) {
    # Iterate over each element in the column
    for (i in 1:length(df[[col]])) {
      # Remove symbols ><&
      df[[col]][i] <- gsub("[!?~]", NA, df[[col]][i])
    }
  }
  return(df)
}

deepsoil<-NA_symb(deepsoil)
deepsoil<-na.omit(deepsoil)
allsoil<-NA_symb(allsoil)
allsoil<-na.omit(allsoil)

#Saving cleaned data 

directory_path <- "/Users/Jasthecoolbean/Desktop/Year 4/dissertation"

# Writing allsoil to cl_allsoil.csv
file_path <- file.path(directory_path, "cl_allsoil.csv")
write.csv(allsoil, file_path, row.names = FALSE)

# Writing deepsoil to cl_Deeper_soil.csv
file_path_2 <- file.path(directory_path, "cl_Deeper_soil.csv")
write.csv(deepsoil, file_path_2, row.names = FALSE)

#--------------------------------------------------------------------------------------------------------#

# Loading the files
cl_allsoil <- read_csv("/Users/Jasthecoolbean/Desktop/Year 4/dissertation/cl_allsoil.csv")
cl_deepsoil <- read_csv("/Users/Jasthecoolbean/Desktop/Year 4/dissertation/cl_Deeper_soil.csv")

#Check BDL
result_list <- list()  # Create an empty list to store results
cl_allsoil_n<-colnames(cl_allsoil[,5:48])

for (col_name in cl_allsoil_n) {
  result <- sum(grepl("-", cl_allsoil[[col_name]]))
  dt <- data.table(
    column_name = col_name,
    BDL_present = result
  )
  result_list[[length(result_list) + 1]] <- dt
}

final_dt <- rbindlist(result_list)
print(final_dt) # Over 50 BDL: Ag, Bi, Cd, Cs, Mo, Na, Sb, Ta, Te, Tl, W

#Remote below extreme BDL values

cl_allsoil <- cl_allsoil[, !names(cl_allsoil) %in% c("Ag", "Bi", "Cd", "Cs", "Mo", "Na", "Sb", "Ta", "Te", "Tl", "W","Zr","U")]
cl_deepsoil <- cl_deepsoil[, !names(cl_deepsoil) %in% c("Ag", "Bi", "Cd", "Cs", "Mo", "Na", "Sb", "Ta", "Te", "Tl", "W","Zr","U")]

#Specify the column index range you want to check
start_column <- 5  # Replace with the starting column index
end_column <- 35   # Replace with the ending column index

# Function to check for non-numeric symbols in a column
contains_non_numeric <- function(x) any(grepl("[^0-9.-]", x))
non_numeric_columns <- lapply(cl_allsoil[, start_column:end_column, with = FALSE], contains_non_numeric)
columns_with_non_numeric <- (start_column:end_column)[unlist(non_numeric_columns)]
print(columns_with_non_numeric)

#There are issues with Zr and U - there is an A printed (These elements aren't important anyways)
#impute BDL

cl_allsoil<-as.data.table(cl_allsoil)
cl_allsoil<-impute_BDL(cl_allsoil)

cl_deepsoil<-as.data.table(cl_deepsoil)
cl_deepsoil<-impute_BDL(cl_deepsoil)

# Writing allsoil to cl_allsoil.csv

directory_path <- "/Users/Jasthecoolbean/Desktop/Year 4/dissertation"

file_path_3 <- file.path(directory_path, "cl_allsoil.csv")
write.csv(cl_allsoil, file_path_3, row.names = FALSE)

# Writing deepsoil to cl_Deeper_soil.csv
file_path_4 <- file.path(directory_path, "cl_Deeper_soil.csv")
write.csv(cl_deepsoil, file_path_4, row.names = FALSE)

#--------------------------------------------------------------------------------------------------------#

# Loading the files
cl_allsoil <- read_csv("/Users/Jasthecoolbean/Desktop/Year 4/dissertation/cl_allsoil.csv")
cl_deepsoil <- read_csv("/Users/Jasthecoolbean/Desktop/Year 4/dissertation/cl_Deeper_soil.csv")

### How does urban area differ from rural areas in terms of the sources of heavy metals?

# Urban data is taken from mainly 2001, and is assumed to be anthropogenic + background. 
# Rural data is taken from mainly 2010, and is assumed to be background. It is also assumed
  # That the rural area has not changed significantly in terms of heavy metal pollution
  # between 2001 and 2010.

#Urban and rural biplot

selected_years <- c(2001, 2002, 2010,2011)
subset_cl_allsoil <- cl_allsoil[cl_allsoil$Year %in% selected_years, ]

par(mfrow = c(1,2))
urban_rural_plot<-ggplot.biplot(subset_cl_allsoil, vars = setdiff(select.VarsElements(subset_cl_allsoil), c(select.VarsREE(subset_cl_allsoil))),
              color = "Type",
              shape = "landuse",
              Arrow = F, TextRepel = T, arrow.len = 0.04,arrowhead.size = 0.01)
  ggtitle("cleaned top soil between 2001-2002")
  
urban_rural_plot

#selected elements

terms<-c("As", "Cr", "Ni", "Pb", "Zn", "Cu", "Se", "Co", "V", "Type", "landuse","X_COORD","Y_COORD")
ele<-c("As", "Cr", "Ni", "Pb", "Zn", "Cu", "Se", "Co", "V")
elements<-subset_cl_allsoil[,ele]
sel_ele <- subset_cl_allsoil[,terms]

#Linear model

ilrtrans<-ilr(elements)

#alr always have pairwise ratio
#ilr 
linmod<-lm(ilrtrans~Type,data=sel_ele)
anova(linmod)

linmod|> coefficients() |> ilr2clr(x=elements)


ilrBase(x=elements)


par(mfrow = c(1,2))
sel_ele_biplot<-ggplot.biplot(sel_ele, vars = setdiff(select.VarsElements(sel_ele), c(select.VarsREE(sel_ele))),
                                color = "Type",
                                shape = "landuse",
                                Arrow = F, TextRepel = T, arrow.len = 0.04,arrowhead.size = 0.01) 

sel_ele_biplot

clr(elements) |> princomp() |> loadings()
clr(elements) |> princomp() |> screeplot()




#Essentially, the larger the distance, the smaller the correlation

par(mfrow = c(1, 1), mar = c(5, 5, 2, 2))
lr<-sel_ele[,..ele]

geom_mean_Pb<-exp(mean(log(lr$Pb)))
geom_mean_Cu<-exp(mean(log(lr$Cr)))
geom_mean_Cr<-exp(mean(log(lr$Cr)))

x=plot(log((lr$Pb) / geom_mean_Pb) ~ log((lr$Cu / geom_mean_Cu)), cex = 0.1)
cor((lr$Pb / geom_mean_Pb),log(lr$Cu / geom_mean_Cu),method = "spearman")
plot(log((lr$Pb) / geom_mean_Pb) ~ log((lr$Cr / geom_mean_Cr)), cex = 0.1)
cor((lr$Pb / geom_mean_Pb),log(lr$Cr / geom_mean_Cr),method = "spearman")

#Biplot with Cu, Zn and Pb anomalies + filtering into a csv
sel_ele_biplot+geom_vline(xintercept = 0.03, linetype = "dashed", color = "black")
ZnCuPb_anom <- sel_ele_biplot$data[sel_ele_biplot$data$Type == "Urban", ]
ZnCuPb_anom<- ZnCuPb_anom[ZnCuPb_anom$Comp.1 > 0.03, c("X_COORD", "Y_COORD")]%>%as.data.table
write.csv(ZnCuPb_anom, file = "ZnCuPb.csv", row.names = FALSE)

#selecting data points for comparison in bar plot + filtering into a csv

ZnCuPb_normal <- sel_ele_biplot$data[sel_ele_biplot$data$Type == "Urban", ]
ZnCuPb_normal<- ZnCuPb_normal[ZnCuPb_normal$Comp.1 < 0.03, c("X_COORD", "Y_COORD")]%>%as.data.table
set.seed(123)  # Setting a seed for reproducibility
# Randomly select 171 points
ZnCuPb_normal <- ZnCuPb_normal[sample(.N, 165), ]
write.csv(ZnCuPb_normal, file = "ZnCuPb_normal.csv", row.names = FALSE)


#Investigate the outliers in sel_ele_biplot -> what are they?

#Filtering four anomalous points, 1, 2, 3, 4, counting from top
#sel_ele_biplot+geom_vline(xintercept = 0.05, linetype = "dashed", color = "black")+geom_hline(yintercept = -0.05, linetype = "dashed", color = "black")
pt_1_anom <- sel_ele_biplot$data[sel_ele_biplot$data$Comp.1 > 0.06 & sel_ele_biplot$data$Comp.2 > 0.05, c("X_COORD", "Y_COORD")]%>%as.data.table
pt_2_anom <- sel_ele_biplot$data[sel_ele_biplot$data$Comp.1 > 0.082 & sel_ele_biplot$data$Comp.2 > 0.04, c("X_COORD", "Y_COORD")]%>%as.data.table
pt_3_anom <- sel_ele_biplot$data[sel_ele_biplot$data$Comp.1 > 0.082 & sel_ele_biplot$data$Comp.2 < 0.04, c("X_COORD", "Y_COORD")]%>%as.data.table
pt_4_anom <- sel_ele_biplot$data[sel_ele_biplot$data$Comp.1 > 0.05 & sel_ele_biplot$data$Comp.2 < -0.05, c("X_COORD", "Y_COORD")]%>%as.data.table

library(data.table)
anomalous_points <- rbindlist(list(pt_1_anom, pt_2_anom, pt_3_anom, pt_4_anom))
print(anomalous_points)

# Urban biplot
selected_type_u<-"Urban"
urban_subset<- sel_ele[sel_ele$Type %in% selected_type_u,]

par(mfrow = c(1,2))
urban_plot<-ggplot.biplot(urban_subset, vars = setdiff(select.VarsElements(urban_subset), c(select.VarsREE(urban_subset))),
                          color="Type",
                          shape = "landuse",
                          Arrow = F, TextRepel = T, arrow.len = 0.04,arrowhead.size = 0.01)
ggtitle("cleaned top soil between 2001-2002 in URBAN")

urban_plot

#Rural biplot 
selected_type_r<-"Rural"
rural_subset<- sel_ele[sel_ele$Type %in% selected_type_r, ]

par(mfrow = c(1,2))
rural_plot<-ggplot.biplot(rural_subset, vars = setdiff(select.VarsElements(rural_subset), c(select.VarsREE(rural_subset))),
                          color = "Type",
                          shape = "landuse",
                          Arrow = F, TextRepel = T, arrow.len = 0.1,arrowhead.size = 0.01)
ggtitle("cleaned top soil between 2001-2002 in RURAL")

rural_plot

#Heat map for sel_ele

sel_ele[, select.VarsElements(sel_ele), with = F] %>% acomp %>% variation %>% heatmap

#### MINI CONCLUSION: 

# Rural and Urban areas have different spread of data points
    # Their selected metals compositions are different from each other
# There is no clear clustering of landuses
    # Metal compositions between landuses are not that different from each other
    # This could be because major heavy metal pollution occurred ~100 years ago
    # Landuses we see now are different from 100 years ago
# 65% variance from PC1 and PC2 explain the spread of the data
# Heat map: The square is bright between Pb and Zn and Cu, 
  # there is a small log ratio between Pb and Zn and Cu
  # Little variance between Pb and Zn (How they change in relation to each other)
  # Same for V, Co and Cr 
# Heat map: The square is darkest between Pb and Ni, Cr, Co V 
  # there is a large ratio between Pb and Ni, Cr, Co V 
  # Large variance, Lots more Pb in urban or lots less Cr/Co/V in rural

#### Next steps:

#In agriculture, what does the biplot look like? [NULL]

selected_landuse_agri<-"agricultural"
agri_subset<- sel_ele[sel_ele$landuse %in% selected_landuse_agri,]

par(mfrow = c(1,2))
agri_plot<-ggplot.biplot(agri_subset, vars = setdiff(select.VarsElements(agri_subset), c(select.VarsREE(agri_subset))),
                          color = "Type",
                          Arrow = F, TextRepel = T, arrow.len = 0.04,arrowhead.size = 0.01)
ggtitle("cleaned top soil between 2001-2002 in URBAN")

agri_plot

#how about in industrial? [NULL]

selected_landuse_indus<-"industrial"
indus_subset<- sel_ele[sel_ele$landuse %in% selected_landuse_indus,]

par(mfrow = c(1,2))
indus_plot<-ggplot.biplot(indus_subset, vars = setdiff(select.VarsElements(indus_subset), c(select.VarsREE(indus_subset))),
                         color = "Type",
                         Arrow = F, TextRepel = T, arrow.len = 0.04,arrowhead.size = 0.01)
ggtitle("cleaned top soil between 2001-2002 in URBAN")

indus_plot

#What about between landuses? [NULL]

elements<-c("As", "Cr", "Ni", "Pb", "Zn", "Cu", "Se", "Co", "V", "Type", "landuse","X_COORD","Y_COORD")
sel_ele <- subset_cl_allsoil[,elements]

par(mfrow = c(1,2))
landuse_biplot<-ggplot.biplot(sel_ele, vars = setdiff(select.VarsElements(sel_ele), c(select.VarsREE(sel_ele))),
                              color = "landuse",
                              Arrow = F, TextRepel = T, arrow.len = 0.04,arrowhead.size = 0.01,size=0.4) 

landuse_biplot

# The different spread of data between rural and urban is mostly due to 
  #anthropogenic influences, such that perhaps Pb Zn and Cu source are dominating urban land.
  #How have these elemental concentration changed with time?

# To statistically prove that, How to do ANOVA with compositional data?

new_elements<-sel_ele[,..ele]
hi<-clr(new_elements)
hi2<-as.data.table(clr2ilr(hi))
setnames(hi2, colnames(hi))
#new elements has 10 variables and ilr has 9
linmod = lm(ilrdata~Type,data=sel_ele)
anova(linmod)

sel_ele
library(car)
vif(linmod)


pdf("no_comp_pca.pdf", width = 10, height = 15)

#PLotting PCA without transformation
#Reference: https://www.benjaminbell.co.uk/2018/02/principal-components-analysis-pca-in-r.html
data<-sel_ele[,1:11]
ele<-c("As", "Cr", "Ni", "Pb", "Zn", "Cu", "Se", "Co", "V")

# PCA using base function - prcomp()
p <- prcomp(data[,ele], scale=TRUE)
# Summary
s <- summary(p)

# Screeplot
layout(matrix(1:2, ncol=2))
screeplot(p)
screeplot(p, type="lines")

#Create color symbols for urban and rural
pch.group <- ifelse(data$Type == "Urban", 21, 21)
col.group <- ifelse(data$Type == "Urban", "darkturquoise", "coral1")

# Plot individuals
plot(p$x[, 2], p$x[, 3], xlab = paste("PCA 1 (", round(s$importance[2] * 100, 1), "%)", sep = ""), 
     ylab = paste("PCA 2 (", round(s$importance[5] * 100, 1), "%)", sep = ""), 
     pch = pch.group, col = col.group, bg = col.group, cex = 0.8, las = 1, asp = 1)

# Add grid lines
abline(v=0, lty=1, col="black")
abline(h=0, lty=1, col="black")
# Add labels
text(p$x[,2], p$x[,3], labels=row.names(p$x), pos=c(1,3,4,2), font=2)

# Get co-ordinates of variables (loadings), and multiply by 10
l.x <- p$rotation[,2]*40
l.y <- p$rotation[,3]*40

# Draw arrows
arrows(x0=0, x1=l.x, y0=0, y1=l.y, col="black", length=0.15, lwd=1.5)

# Label position
l.pos <- l.y # Create a vector of y axis coordinates
lo <- which(l.y < 0) # Get the variables on the bottom half of the plot
hi <- which(l.y > 0) # Get variables on the top half
# Replace values in the vector
l.pos <- replace(l.pos, lo, "1")
l.pos <- replace(l.pos, hi, "3")

# Variable labels
text(l.x, l.y, labels=row.names(p$rotation), col="black", pos=l.pos)

# Add the legend with the desired properties
legend("topleft", legend = c("Urban", "Rural"), col = "black", pt.bg = c("darkturquoise", "coral1"), pch = 21, pt.cex = 1.5)

# Extract scores from the PCA result
scores <- p$x
pc1<-scores[,1]

# Display the first few rows of the scores
head(scores)

dev.off()

# Get PCA loadings
loadings_table <- as.data.frame(p$rotation)
colnames(loadings_table) <- paste("PC", 1:ncol(loadings_table), sep = "_")

# Print the loadings table
print(loadings_table)

#

lm()

#MANOVA

How do I run MANOVA on this?

#--------------------------------------------------------------------------------------------------------#

# Pb, Zn and Cu changes with time 
# I have expanded the 2018 area beyond the points, so that it encompasses a larger area
# This way, data points from 2001 and 2010 can fall into the area to be assessed.
# However, even at 1000m distance, there are only 4 data points present for 2010.
# I am only going to look at 2010 vs 2018 for the time series change

SWglasgow <- read_csv("/Users/Jasthecoolbean/Desktop/Year 4/dissertation/cl_allsoil_time_series.csv")

# Assuming your columns are named Pb, Cu, and Zn
SWglasgow_long <- SWglasgow %>%
  gather(Element, Concentration, Pb, Cu, Zn,)

# Filter for specific years
years_of_interest <- c(2001, 2018)
SWglasgow_filtered <- SWglasgow_long %>% 
  filter(Year %in% years_of_interest)

# Create a box plot
ggplot(SWglasgow_filtered, aes(x = as.factor(Year), y = Concentration, fill = Element)) +
  geom_boxplot(position = "dodge") +
  labs(title = "Time Series Box Plot of Concentration",
       x = "Year",
       y = "Concentration") +
  scale_fill_manual(values = c("Pb" = "red", "Cu" = "blue", "Zn" = "green")) +
  theme_minimal()

# Create a boxplot for compositional changes 

Pbs<-log(SWglasgow$Pb/SWglasgow$Sr)
Cus<-log(SWglasgow$Cu/SWglasgow$Sr)
Zns<-log(SWglasgow$Zn/SWglasgow$Sr)

table<-as.data.table(cbind(Pbs,Cus,Zns,Year=SWglasgow$Year))

# Assuming your columns are named Pb, Cu, and Zn
SWglasgow_longer <- table %>%
  gather(Element, Concentration, Pbs, Cus, Zns)

ggplot(SWglasgow_filter, aes(x = as.factor(Year), y = Concentration, fill = Element)) +
  geom_boxplot(position = "dodge") +
  labs(title = "Time Series Box Plot of log ratio",
       x = "Year",
       y = "log(element / Sr)") +
  scale_fill_manual(values = c("Pbs" = "red", "Cus" = "blue", "Zns" = "green")) +
  theme_minimal()

# Perform separate ANOVA tests for each element
anova_result_Pbs <- aov(Pbs~Year,data=table)
anova_result_Cus <- aov(Cus~Year,data=table)
anova_result_Zns <- aov(Zns~Year,data=table)

# Display summaries for each ANOVA test
summary(anova_result_Pbs)
summary(anova_result_Cus)
summary(anova_result_Zns)

# 42 data points for 2001, 4 data points from 2011, 47 data points from 2018

#Comment on bad data in, bad data out.

#What are the driving factors? 

#--------------------------------------------------------------------------------------------------------#

# Driving factor 1: Distance

sel_ele_dis<-read_csv("/Users/Jasthecoolbean/Desktop/Year 4/dissertation/Distance_from_Clyde.csv")

# Will distance from river clyde generate different groupings in 
 
sel_ele_dis <- as_tibble(sel_ele_dis) %>%
  select(-HubName) %>%
  mutate(
    Distance_from_River_Clyde_meters = case_when(
      between(HubDist, 0, 2000) ~ ">2000",
      #between(HubDist, 2000, 6000) ~ "2000-6000",
      #between(HubDist, 6000,10000) ~ "6000-10000",
      #between(HubDist, 10000, 25000) ~ "10000-25000",
      between(HubDist, 25000, 50000) ~ "25000-50000",
      TRUE ~ NA_character_
    )
  ) %>%
  select(-landuse, -X_COORD, -Y_COORD, -HubDist) 

sel_ele_dis<-na.omit(sel_ele_dis)


par(mfrow = c(1,2))
sel_ele_dis_biplot<-ggplot.biplot(sel_ele_dis, vars = setdiff(select.VarsElements(sel_ele_dis), c(select.VarsREE(sel_ele_dis))),
                              color = "Distance_from_River_Clyde_meters",
                              Arrow = F, TextRepel = T)
ggtitle("cleaned top soil selected elements distance from clyde between 2001-2002")

sel_ele_dis_biplot

#How about distance to road?

rd_dis<-read_csv("/Users/Jasthecoolbean/Desktop/Year 4/dissertation/Distance_to_nearest_road.csv")
filtered_rd_dis <- rd_dis %>%
  filter((Year == 2001 | Year == 2002))
elements1<-c("As", "Cr", "Ni", "Pb", "Zn", "Cu", "Se", "Co", "V","Sr")
filtered_rd_dis_1 <- filtered_rd_dis[,elements1]
alr_dis<-as.data.table(alr(filtered_rd_dis_1))
alr_dis_bind <- cbind(alr_dis, Distance_to_road=filtered_rd_dis$HubDist)

#Linmod for all alr elements

response_columns <- c("As", "Cr", "Ni", "Pb", "Zn", "Cu", "Se", "Co", "V")

# Create an empty list to store the linear regression models
linmod_list <- list()

# Loop through each response variable
for (col in response_columns) {
  formula <- as.formula(paste(col, "~ Distance_to_road"))
  linmod <- lm(formula, data = alr_dis_bind)
  linmod_list[[col]] <- linmod
}

# Access the summary of each linear regression model
for (col in response_columns) {
  print(paste("Summary for", col, ":"))
  print(summary(linmod_list[[col]]))
}


library(ggplot2)

# Assuming response_columns contains the variable names
response_columns <- c("As", "Cr", "Ni","Pb", "Zn", "Cu", "Se", "Co", "V")

# Create a list to store ggplot objects
plot_list <- list()

# Loop through each response variable and create ggplot
for (col in response_columns) {
  formula <- as.formula(paste(col, "~ Distance_to_road"))
  linmod <- lm(formula, data = alr_dis_bind)
  
  plot <- ggplot(alr_dis_bind, aes(y = get(col), x = Distance_to_road)) +
    geom_point() +
    geom_smooth(method = "lm") +
    labs(x = "Distance from road (m)", y = paste("log(", col, "/Sr)")) +
    ggtitle(paste("Linear Regression for", col))
  
  plot_list[[col]] <- plot
}
# Combine all plots into a single panel using facet_wrap
combined_plot <- do.call(gridExtra::grid.arrange, c(plot_list, ncol = 2))

# Print or display the combined plot
print(combined_plot)

#Just Pb

PbPlot <- ggplot(alr_dis_bind, aes(y = Pb, x = Distance_to_road)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(x = "Distance from road (m)", y = expression(log(Zn/Sr)))

PbPlot


test<-lm(Pb~Distance_to_road,data=alr_dis_bind)
plot(test)

#distributional test: test normality - shapiro wilks test 
shapiro.test(test$residuals)
#distribution test have opposite of hypothesis null is: no difference between model residuals and residuals of a normal distribution
#we want very high p-value, our value is too low 
(test)
#Tukey transformation to fix the distribution Nudge pvalue above 0.05
# if above 0.5 it has high leverage, it is fine (all assumptions are met)



#equal variance
#normal distribution
#little bit of flat red line

#--------------------------------------------------------------------------------------------------------#

#Investigating how Princomp 1 changes on map

sel_ele_w_coord<-cl_allsoil[,c("As", "Cr", "Ni", "V","Pb", "Zn", "Cu", "Type", "landuse","X_COORD","Y_COORD")]

file_path_3 <- file.path(directory_path, "sel_ele_qgis.csv")
write.csv(sel_ele_w_coord, file_path_3, row.names = FALSE)

Y=sel_ele_w_coord$Y_COORD
X=sel_ele_w_coord$X_COORD

plot(X,Y,frame.plot=FALSE,xaxt="n",yaxt="n",xlab="",ylab="",type="n")
SymbLegend(X,Y,sel_ele_biplot$data$Comp.1,type="percentile",qutiles<-c(0,0.05,0.25,0.75,0.95,1),symbtype="EDA",symbmagn=0.8,
           leg.position="bottomleft",leg.title="PC1 scores",leg.just="right")

# More dense crosses / higher PC1 scores in the middle (urban area)
# Could this be because more data is collected for the urban area?




