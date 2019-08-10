# ai-package

In Pipfile

my_deep_learning_lib = {git = "https://github.com/Wen-Jian/ai-package.git"}

install package

For training

from MY_DEEP_LEARNING_LIB.heigh_resolution_generator_module import HeighResolutionGenerator 


generator = HeighResolutionGenerator(datasets, batch_size, input_shape=(180, 320), output_shape=(360, 640), channel_size=3, model_name='srcnn_2x')
generator.build_model()
generator.train()


For predit
pred = generator.predit([x_img], (360, 640), 3)
