library(ggplot2)
library(ggthemes)

x = read.csv("perf.txt")
x$disk = ifelse(x$disk, "Data Streamed", "Data In Memory")
x$rank = paste(x$rank)

g = ggplot(x, aes(thread, maxkwps, color=rank)) + 
  geom_point() + 
  geom_line() +
  scale_color_tableau() + 
  theme_bw() + 
  facet_grid( ~ disk) + 
  labs(color="MPI Ranks") +
  xlab("Number of Threads") + 
  ylab("Peak Thousands of Words/Sec Achieved") + 
  ggtitle("w2v() Benchmark on text8 Data") +
  scale_x_continuous(breaks = seq(2, 6, by = 2))

ggsave(g, file="perf.png")
