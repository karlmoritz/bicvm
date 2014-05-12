#############################################
# Libraries
#############################################

add_library(modflattree SHARED backpropagator.cc recursive_autoencoder.cc singleprop.cc)
target_link_libraries(modflattree dualtrain)
