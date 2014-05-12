#############################################
# Libraries
#############################################

set(LBFGS_FLOAT "32")
add_library(basemod SHARED recursive_autoencoder.cc singlepropbase.cc backpropagatorbase.cc)
add_library(dualtrain SHARED train_update.cc)
target_link_libraries(dualtrain basemod)
add_library(basics SHARED fast_math.cc finetune_classifier.cc reindex_dict.cc
  senna.cc shared_defs.cc utils.cc)
add_library(loaders SHARED load_alpino.cc load_ccg.cc load_doc.cc load_plain.cc
  load_stanford.cc)
target_link_libraries(loaders ${Boost_FILESYSTEM_LIBRARIES}
  ${Boost_SYSTEM_LIBRARIES})
add_library(trainers SHARED finite_grad_check.cc train_adagrad.cc train_sgd.cc)

# backpropagatorbase.cc   finite_grad_check.cc  load_doc.cc       recursive_autoencoder.cc  shared_defs.cc     train_lbfgs.cc   train_vsgd.cc
# fast_math.cc            load_alpino.cc        load_plain.cc     reindex_dict.cc           singlepropbase.cc  train_sgd.cc     utils.cc
# finetune_classifier.cc  load_ccg.cc           load_stanford.cc  senna.cc                  train_adagrad.cc   train_update.cc


#############################################
# Executables
#############################################

# add_executable(train_cnlm train_cnlm.cc cnlm.cc additive-cnlm.cc)
# set(EXECUTABLES train_cnlm cg perplexity label printconfig print_cldc finite_gradient_check)
# foreach(f ${EXECUTABLES})
  # add_executable(${f} ${f}.cc)
  # target_link_libraries(${f} cnlm)
# endforeach(f)
