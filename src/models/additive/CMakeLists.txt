#############################################
# Libraries
#############################################

add_library(modadditive SHARED backpropagator.cc recursive_autoencoder.cc singleprop.cc)
target_link_libraries(modadditive dualtrain)
