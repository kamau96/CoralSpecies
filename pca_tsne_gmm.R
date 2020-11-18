# Front matter
rm(list=ls())
library(data.table)
library(Rtsne)
library(ggplot2)
library(caret)
library(ClusterR)

# Load in data 
train <- fread("./speciesdata.csv")
submission <- fread("./exampleSubmission.csv")
train_id<-train$id
train[] <- lapply(train, as.character)

#Converting the labels in the i.e locus into binary variables

dummies <- dummyVars(id~., data = train)
train <- predict(dummies, newdata = train)
train<- data.table(train)

# Running a pca
pca <- prcomp(train,center=TRUE,scale = TRUE)

# Look at the percent variance explained by each pca
screeplot(pca)

# Look at the rotation of the variables on the PCs
pca

# Loot at the values of the scree plot in a table 
summary(pca)

# Look at the biplot of the first 2 PCs
biplot(pca)

# Project the data into PCA space
pca_dt <- data.table(pca$x)


# Plot with the party data 
ggplot(pca_dt,aes(x=PC1,y=PC2)) + geom_point()

# Run t-SNE (NOTE: pca is built into Rtsne)
set.seed(3)
#5 no clear separation
#10 no clear separation
#15 no clear separation
#30 start seeing separation
#45 three clear separation
#60 separation lost
#80 separation lost
#100 separation lost
perplexity.value <- 45
tsne <- Rtsne(pca_dt, dims=3, pca = T, perplexity=perplexity.value, check_duplicates = F)

# Grab tSNE coordinates
tsne_dt <- data.table(tsne$Y)
ggplot(tsne_dt, aes(x=V1,y=V2,V3)) + geom_point() + labs(title = paste("perplexity = ", perplexity.value))


# Finding optimal K (i.e. number of clusters) with GMM (i.e. gaussian mixture models)
k_bic <- Optimal_Clusters_GMM(tsne_dt[,.(V1,V2,V3)],max_clusters = 10,criterion = "BIC")
delta_k <- c(NA,k_bic[-1] - k_bic[-length(k_bic)]) #change in model fit

del_k_tab <- data.table(delta_k=delta_k, k=1:length(delta_k)) #for clear visualization
ggplot(del_k_tab,aes(x=k,y=-delta_k)) + geom_point() + geom_line() +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) +
  geom_text(aes(label=k),hjust=0, vjust=-1)

opt_k <- 3

# Run GMM with our chosen optimal k value
gmm_data <- GMM(tsne_dt[,.(V1,V2,V3)],opt_k)

# Converting log-likelihood into probability
l_clust <- gmm_data$Log_likelihood^10
l_clust <- data.table(l_clust)
net_lh <- apply(l_clust,1,FUN=function(x){sum(1/x)})
cluster_prob <- 1/l_clust/net_lh

# To see how cluster 1 look like
tsne_dt$Cluster_1_prob <- cluster_prob$V1
ggplot(tsne_dt,aes(x=V1,y=V2,col=Cluster_1_prob)) + geom_point()

# Assigning hard labels
tsne_dt$gmm_labels <- max.col(cluster_prob, ties.method = "random")
ggplot(tsne_dt,aes(x=V1,y=V2,col=gmm_labels)) + geom_point()
table(train_id, tsne_dt$gmm_labels)

# Saving the figures
plots.dir.path <- list.files(tempdir(), pattern="rs-graphics", full.names = TRUE)
plots.png.paths <- list.files(plots.dir.path, pattern=".png", full.names = TRUE)
submission$species1<-cluster_prob$V3
submission$species2<-cluster_prob$V1
submission$species3<-cluster_prob$V2
fwrite(submission,"./exampleSubmission.csv")








