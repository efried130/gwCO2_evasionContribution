#install.packages('phreeqc')
library(phreeqc)
library(dplyr)
library(ggplot2)
# phreeqc::phrAccumulateLine		Accumulate line(s) for input to phreeqc.
# phreeqc::phrGetOutputStrings		Retrieve standard phreeqc output.
# phreeqc::phrLoadDatabase		Load a phreeqc database file
# phreeqc::phrLoadDatabaseString		Load a phreeqc database as a string or a list of strings.
# phreeqc::phrRunFile		Run phreeqc input file
# phreeqc::phrRunString		Runs phreeqc using the given string as input.
# phreeqc::phreeqc-package		R interface to the PHREEQC geochemical modeling program.
# phreeqc::phreeqc.dat		The phreeqc.dat database


#Now do this for gas pressure CO2

calc_co2 <- function(input_filePath){
  # turn on output
  phrLoadDatabaseString(phreeqc.dat)
  phrSetOutputFileOn(TRUE)
  phrSetOutputFileName(file.path("/Users/elisafriedmann/Library/CloudStorage/OneDrive-UniversityofMassachusetts/UMass/CO2/phreeqc/", "co2_speciation.sel"))
  # run input
  phrRunFile(input_filePath)
  output <- phrGetSelectedOutput()
  df <- output[[1]]
  #print(df)
  return(df)
}
input_filePath = "/Users/elisafriedmann/Library/CloudStorage/OneDrive-UniversityofMassachusetts/UMass/CO2/phreeqc/phreeqcInput4.txt"
phreeqOutput <-calc_co2(input_filePath)
df <- phreeqOutput
df
summary(df$CO2_ppm)
#df2csv <- write_csv(df, file = ("/Users/elisafriedmann/Library/CloudStorage/OneDrive-UniversityofMassachusetts/UMass/CO2/phreeqc/co2_calcs_phreeqc.csv"))


df <- df %>%
  mutate(CO2_ppm = as.numeric(CO2_ppm)) %>%
  filter(CO2_ppm < 100000) %>%
  filter(CO2_ppm > 1)
hist(df$CO2_ppm, breaks=100)
summary(df$CO2_ppm)
print(mean(df$CO2_ppm))
nrow(df)

p <- ggplot(df, aes(CO2_ppm)) +
  geom_histogram(binwidth =100, fill='skyblue', color = 'black') +
  geom_vline(xintercept = mean(df$CO2_ppm), linetype= 'dashed', color = 'black') +
  geom_vline(xintercept = median(df$CO2_ppm), linetype= 'dotted', color = 'black') +
  annotate("text", x=mean(df$CO2_ppm), y= 1500, label = paste("Mean (ppm):", round(mean(df$CO2_ppm), 2)), vjust = -0.2, hjust=-1.0,color='black') +
  annotate("text", x=median(df$CO2_ppm), y= 1500, label = paste("Median (ppm):", round(median(df$CO2_ppm), 2)), vjust = 1.0, hjust = -1.2, color='black') +
  labs(x= 'pCO2 (ppm)', y = 'Frequency')
p

#Check work
ratio <- df$CO2_ppm/10^6/df$molality_CO2
plot(df$temp.C., ratio)




