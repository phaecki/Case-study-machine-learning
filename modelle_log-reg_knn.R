# Pakete laden
library(plyr)
library(tidyverse)
library(readr)
library(caret)
library(gridExtra)
library(pROC)

# CSV-Dateien in Variablen laden
bloodtrain <- read_csv("bloodtrain.csv")
bloodtest <- read_csv("bloodtest.csv") # Zukunfts-Datenset

# Exploration
## Class herausfinden
sapply(bloodtrain, class)
sapply(bloodtest, class)

## Datentyp ermitteln
head(bloodtrain)
head(bloodtest)

## Summary Statistics erstellen und Verteilungen herausfinden
summary(bloodtrain)
summary(bloodtest)

## Struktur ermitteln
str(bloodtrain)
str(bloodtest)

## Alternativ «glimpse» verwenden anstelle von «str»
glimpse(bloodtrain)
glimpse(bloodtest)

## Daten von bloodtrain visualisieren, Verteilungen ansehen
boxplot(bloodtrain)

## Streu-Martix mit Histogrammen & Trendlinien, Zusammenhänge herausfinden
panel.hist <- function(x, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5) )
  h <- hist(x, plot = FALSE)
  breaks <- h$breaks; nB <- length(breaks)
  y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = "cyan", ...)
}
pairs(bloodtrain[1:6], main="Streu-Matrix für Traing-Datensatz", panel = panel.smooth,
      cex = 1, pch = 22, bg = "light blue",
      diag.panel = panel.hist, cex.labels = 1.5, font.labels = 2)
pairs(bloodtest[1:5], main="Streu-Matrix für Test-Datensatz", panel = panel.smooth,
      cex = 1, pch = 22, bg = "light blue",
      diag.panel = panel.hist, cex.labels = 1.5, font.labels = 2)

## Berechnen der Korrelation von «Number of Donations» und «Total Volume Donated (c.c.)»
cor(bloodtrain$`Number of Donations`, bloodtrain$`Total Volume Donated (c.c.)`)

## Analyse «Months since Last Donation», Verteilung der Daten prüfen
hist(bloodtrain$`Months since Last Donation`, breaks = 60)

# Vorbereitung der Daten
## Umbenennen der Variablen
df_btrain <- bloodtrain %>% 
  rename(Id = X1,
         Recency = `Months since Last Donation`,
         Frequency = `Number of Donations`,
         Volume = `Total Volume Donated (c.c.)`,
         DonationPeriod = `Months since First Donation`,
         diddonate = `Made Donation in March 2007`)

df_btest <- bloodtest %>% 
  rename(Id = X1,
         Recency = `Months since Last Donation`,
         Frequency = `Number of Donations`,
         Volume = `Total Volume Donated (c.c.)`,
         DonationPeriod = `Months since First Donation`)

## Prüfen, ob leere Werte vorhanden
sum(is.na(df_btrain$Recency))
sum(is.na(df_btrain$Frequency))
sum(is.na(df_btrain$Volume))
sum(is.na(df_btrain$DonationPeriod))
sum(is.na(df_btrain$diddonate))

## Prüfen, ob doppelte Einträge vorhanden
sum(df_btrain[duplicated(df_btrain) == TRUE,])

## Extreme Werte prüfen, interaktiv bestimmen
par(mfrow=c(1,1))
plot(x = df_btrain$DonationPeriod, y = df_btrain$Recency, main = "Zusammenhang für Training-Daten", 
     xlab = "Donation-Periode", ylab = "Recency")
extrem_train <- identify(x = df_btrain$DonationPeriod, y = df_btrain$Recency, tolerance = 0.25)
### Achtung: gibt Zeilen-Nummer in der Excel-Datei - 1 an (wg Header)
extrem_train
df_btrain[extrem_train,]

### Entscheid: Extreme Werte werden nicht gelöscht.

## Variablen/Feature-Auswahl
### Entscheid: Variable «Volume» wird für Modellierung nicht verwendet, 
### da sie mit der Variablen «Frequency» perfekt korreliert.
df_btrain <- df_btrain %>% 
  select(Id, Recency, Frequency, DonationPeriod, diddonate)

### Startwert/set.seed für die Reproduzierbarkeit auswählen
set.seed(123)

# Modellierung
## Datenset in Dataframe konvertieren
df_btrain <- data.frame(df_btrain)

## Variable «diddonate» in kategorische Daten umwandeln
df_btrain$diddonate <- as.factor(df_btrain$diddonate)

## Daten in Training- & Test-/Validierung-Datensatz teilen
partition <- createDataPartition(df_btrain[,1], times = 1, p = 0.75, list = FALSE)
train <- df_btrain[partition, ] # Trainings-Set
test <- df_btrain[-partition, ] # Test-/Validierungs-Set

## Logistische Regression

### Dummy Data set erstellen
dummies = dummyVars(diddonate ~ Recency + Frequency + DonationPeriod, data = train, 
                    fullRank = TRUE)

### fullRank = TRUE macht aus einem Factor mit n levels n-1 dummy variabeln
train_dummies = data.frame(predict(dummies, newdata = train))
test_dummies = data.frame(predict(dummies, newdata = test))

### Struktur ermitteln
str(train_dummies)

### Parameter glm ausgeben
params <- getModelInfo("glm")
params$glm$parameters

### Modell erstellen
log_regression <- train(y=train$diddonate, x=train_dummies, method="glm", family = "binomial")

### Log_regression ausgeben
log_regression

### Vorhersage für Test-Daten
score <- predict(log_regression, newdata = test_dummies)
confusionMatrix(score, test$diddonate)

### Vorhersage für Test-Daten mit type = "prob" für Einreichung
logreg_pred_prob <- predict(log_regression, newdata = df_btest, type = "prob")

### Einreichung vorbereiten
logreg_submission <- data.frame(df_btest$Id, logreg_pred_prob$`1`)
colnames(logreg_submission) <- c(" ", "Made Donation in March 2007")
write_csv(logreg_submission, "logreg_submission.csv")

## kNN

### Daten balanced?
table(train$diddonate)

#### Verhältnis von 1:3.2 ist zwar unausgewogen, aber nicht beunruhigend.

### Parameter ausgeben
params <- getModelInfo("knn")
params$knn$parameters

### Levels des Faktors «diddonate» umbenennen
train$diddonate <- revalue(train$diddonate, c("0" = "N", "1" = "Y"))
test$diddonate <- revalue(test$diddonate, c("0" = "N", "1" = "Y"))

### Training-Kontrollen für wiederholte Kreuz-Validierung
knn_grid <- expand.grid(k = 1:40)
knn_control <- trainControl(method = "repeatedcv", number = 5, classProbs = TRUE, 
                             summaryFunction = mnLogLoss, savePredictions = TRUE)

### kNN-Modell trainieren
knn_model <- train(diddonate ~ Recency + Frequency + DonationPeriod, method = "knn", data = train, preProc=c("center", "scale"), 
                   metric = "logLoss", maximize = FALSE, trControl = knn_control, 
                   tuneGrid = knn_grid)

#### Hinweis: Zentrieren und Skalieren anwenden für Optimierung des Modells

### kNN-Modell ausgeben
knn_model

### Vorhersage für Test-Daten
knn_pred <- predict(knn_model, newdata = test)
confusionMatrix(knn_pred, test$diddonate)

### ROC-Kurve ausgeben
selectedIndices <- knn_model$pred$k == 39
plot.roc(knn_model$pred$obs[selectedIndices],
         knn_model$pred$Y[selectedIndices])

### Vorhersage für Test-Daten mit type = "prob" für Einreichung
knn_pred_prob <- predict(knn_model, newdata = df_btest, type = "prob")

### Einreichung vorbereiten
knn_submission <- data.frame(df_btest$Id, knn_pred_prob$Y)
colnames(knn_submission) <- c(" ", "Made Donation in March 2007")
write_csv(knn_submission, "knn_submission.csv")
