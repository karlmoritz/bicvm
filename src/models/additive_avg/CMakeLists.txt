#############################################
# Libraries
#############################################

add_library(modadditive_avg SHARED backpropagator.cc recursive_autoencoder.cc singleprop.cc)
target_link_libraries(modadditive_avg dualtrain)
