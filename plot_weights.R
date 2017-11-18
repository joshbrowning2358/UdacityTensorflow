library(data.table)
library(ggplot2)

conv1_weights = fread('conv1_weights.csv')
conv1_weights[, layer := rep(1:16, each=5)]
conv1_weights[, row := rep(1:5, times=16)]
to_plot = melt(conv1_weights, id.vars=c('layer', 'row'), variable.name='column', value.name='weight')
to_plot[, column := as.numeric(gsub('V', '', column))]

ggplot(to_plot, aes(x=row, y=column, fill=weight)) + geom_tile() +
    facet_wrap( ~ layer)
