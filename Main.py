class Model:
  def __init__(self, seq_length, vocab_size, batch_size=32, embedding_dim=254, rnn_units=1024):
    # Optimization parameters:
    self.seq_length = seq_length  # Experiment between 50 and 500
    self.batch_size = batch_size

    # Model parameters: 
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim 
    self.rnn_units = rnn_units
    

  def LSTM(self): 
    return tf.keras.layers.LSTM(
      self.rnn_units, 
      return_sequences=True, 
      recurrent_initializer='glorot_uniform',
      recurrent_activation='sigmoid',
      stateful=True,
    )
  

  def build_model(self):
    '''
    Function to build the model, here we create the model, define which activation functions we are going to use and how many layers the model has
    vocab_size = (int) length of vocab
    embedding_dim = (int) embedding dimmension
    rnn_units = (int) rnn units
    batch_size = (int) batch size

    return 
    model = (tensorflow.python.keras.engine.sequential.Sequential) model
    '''
    #Create the sequential model
    model = tf.keras.Sequential()
    #Add the 1. layer, embedding layer to transform indices into dense vectors
    model.add(tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, batch_input_shape=[self.batch_size, None]))
    #Add the 2. layer, LSTM with `rnn_units` number of units. 
    model.add(self.LSTM())
    #Add the 3. layer, Dense (fully-connected) layer that transforms the LSTM output into the vocabulary size. 
    model.add(tf.keras.layers.Dense(self.vocab_size, activation="softmax"))
    print(model.summary())
    return model
  
  def train_step(self, x, y, model, learning_rate=0.001):
    '''
    Function that defines a step of training
    x = (numpy.ndarray) vectorized letters (randomly selected)
    y = (numpy.ndarray) vectorized letters (GT)

    return
    loss = (tensorflow.python.framework.ops.EagerTensor) loss for each batch on a specific t
    '''
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    with tf.GradientTape() as tape:
      y_hat = model(x)
      loss = self.compute_loss(y, y_hat)

    # Now, compute the gradients 
    grads = tape.gradient(loss, model.trainable_variables)
  
    # Apply the gradients to the optimizer so it can update the model accordingly
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

  def compute_loss(self,GT, pred):
    '''
    Function to calculate the loss function. In particular we are using crossentropy loss (negative log likelihood loss)
    GT = (numpy.ndarray) array with the groud truth
    pred = (tensorflow.python.framework.ops.EagerTensor) array with the prediction 

    return
    loss = (tensorflow.python.framework.ops.EagerTensor) loss for each batch on a specific t
    '''
    loss = tf.keras.losses.sparse_categorical_crossentropy(GT, pred, from_logits=True) 
    return loss

  def train(self, model, num_training_iterations, string, checkpoint_prefix = ".",):
    history = []
    print(num_training_iterations)
    for iter in range(num_training_iterations):
      # Grab a batch and propagate it through the network
      x_batch, y_batch = get_batch(string, self.seq_length, self.batch_size)
      loss = self.train_step(x_batch, y_batch, model)

      # Update the progress bar
      history.append(loss.numpy().mean())
      plt.plot(history)
      plt.xlabel("epochs")
      plt.ylabel("error")
      display.clear_output(wait=True)
      display.display(plt.gcf())

      # Update the model with the changed weights!
      if iter % 100 == 0:     
        model.save_weights(checkpoint_prefix)
        
    # Save the trained model and the weights
    model.save_weights(checkpoint_prefix)
    return model, history
  
  def Predict(self, checkpoint_dir = "."):
    model = self.build_model()
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
    return model


