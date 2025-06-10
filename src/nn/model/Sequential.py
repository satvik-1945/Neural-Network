class Sequential():
    def __init__(self):
        self.layers = []
        self.compiled = False
        self.optimizer = None
        self.loss_function = None
        self.metrics = []
 
    def add(self, layer):
        self.layers.append(layer)
 
    def compile(self, optimizer='adam',  metrics=['accuracy']):
        self.optimizer = optimizer
        self.metrics = metrics
        self.compiled = True
 
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'units'):#to make sure we don't make weights matrix of input layer
                if i == 0:  # first layer for input shape
                    if hasattr(layer, 'input_shape'):
                        layer.initialize_weights(layer.input_shape)
                else:  # subsequent layer already knows the previous layer
                    prev_layer = self.layers[i - 1]
                    if hasattr(prev_layer, 'units'):
                        layer.initialize_weights(prev_layer.units)
                    elif hasattr(prev_layer, 'output_size'):
                        layer.initialize_weights(prev_layer.output_size)
 
    def _forward_pass(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
 
    def _backward_pass(self, x, y_true, y_pred, learning_rate):
        grad = self._loss_gradient(y_pred, y_true)
 
        activations = [x]
        current_input = x
        for layer in self.layers:
            current_input = layer.forward(current_input)
            activations.append(current_input)
 
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if hasattr(layer, 'backward'):
                grad = layer.backward(activations[i], grad, learning_rate)
 
    def _calculate_loss(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
 
        if len(y_true.shape) == 1:
            m = y_true.shape[0]
            loss = -np.sum(np.log(y_pred[np.arange(m), y_true])) / m
        else:
            loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
 
        return loss
 
    def _loss_gradient(self, y_pred, y_true):
        m = y_true.shape[0]
        if len(y_true.shape) == 1:
            y_one_hot = np.zeros_like(y_pred)
            y_one_hot[np.arange(m), y_true] = 1
            y_true = y_one_hot
 
        return (y_pred - y_true) / m
 
    def _calculate_accuracy(self, y_pred, y_true):
        predictions = np.argmax(y_pred, axis=1)
        if len(y_true.shape) == 1:
            return np.mean(predictions == y_true)
        else:
            true_labels = np.argmax(y_true, axis=1)
            return np.mean(predictions == true_labels)
 
 
    def fit(self, x_train, y_train, epochs=1, learning_rate=0.001, batch_size=32):
        if not self.compiled:
            raise ValueError("Model must be compiled before training")
 
        n_samples = x_train.shape[0]
        n_batches = n_samples // batch_size
 
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
 
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
 
            print(f"Epoch {epoch + 1}/{epochs}")
 
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
 
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
 
                # Forward pass
                predictions = self._forward_pass(x_batch)
 
                batch_loss = self._calculate_loss(predictions, y_batch)
                batch_accuracy = self._calculate_accuracy(predictions, y_batch)
 
                # Backward pass
                self._backward_pass(x_batch, y_batch, predictions, learning_rate)
 
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
 
                if (batch_idx + 1) % 100 == 0 or batch_idx == n_batches - 1:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    avg_accuracy = epoch_accuracy / (batch_idx + 1)
                    progress = (batch_idx + 1) / n_batches
                    bar_length = 20
                    filled_length = int(bar_length * progress)
                    bar = '━' * filled_length + ' ' * (bar_length - filled_length)
                    print(f"\r{batch_idx + 1}/{n_batches} {bar} {progress:.0%} - "
                          f"accuracy: {avg_accuracy:.4f} - loss: {avg_loss:.4f}", end='')
 
            final_loss = epoch_loss / n_batches
            final_accuracy = epoch_accuracy / n_batches
            print(f"\r{n_batches}/{n_batches} {'━' * 20} 100% - "
                  f"accuracy: {final_accuracy:.4f} - loss: {final_loss:.4f}")
            print()  
    def save(self,filepath):
        model_data = {
            'layers':self.layers,
            'optimizer':self.optimizer,
            'metrics':self.metrics,
            'compiled':self.compiled
        }
 
        with open(filepath, 'wb')as f:
            pickle.dump(model_data,f)
            print(f"Model saved to {filepath}")
 
    def predict(self, x):
        return self._forward_pass(x)