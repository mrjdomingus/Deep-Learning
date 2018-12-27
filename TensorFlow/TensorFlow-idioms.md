# Some useful TensorFlow idioms

## Debugging idioms

Also see: [How to get started debugging TensorFlow](https://medium.freecodecamp.org/debugging-tensorflow-a-starter-e6668ce72617)

**Print tf.Session run-results**

```
import tensorflow as tf
a = tf.placeholder(tf.float32, shape = (5,), name='a-input')
b = tf.square(a, name="b-squares")
sess = tf.InteractiveSession()
print(sess.run(b, feed_dict={a: [1,2,3,4,5]}))
```
Prints: `[ 1.  4.  9. 16. 25.]`

Alternatively, use `Tensor.eval()`:

```
print(b.eval(feed_dict={a: [1,2,3,4,5]},session=sess))
```
Prints: `[ 1.  4.  9. 16. 25.]`

**Embed tf.print in the TensorFlow Graph**

Note: tf.Print (capital P) is deprecated

```
sess = tf.Session()
with sess.as_default():
    a = tf.placeholder(tf.float32, shape = (5,), name='a-input')
    b = tf.square(a, name='b-squares')
    # Create an operator that prints the specified inputs to a desired output stream or logging level.
    # Note: output of debug_prints will go to console, NOT to Jupyter notebook
    debug_prints = tf.print("Tensor values:", {'b': b, 'a': a}, output_stream=sys.stdout)
    # Specify the debug_prints operator as control dependency for tensor c
    with tf.control_dependencies([debug_prints]):
        c = tf.reduce_mean(b, name='c-mean') # Tensor c will only run AFTER debug_prints has executed
    # Now run the session for tensor c
    sess.run([c], feed_dict={a: [1,2,3,4,5]})
 ```
