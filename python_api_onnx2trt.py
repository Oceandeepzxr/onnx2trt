import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def build_engine(model_path):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 20
        builder.max_batch_size = 1
        with open(model_path, "rb") as f:
            parser.parse(f.read())
        engine = builder.build_cuda_engine(network)
        return engine


if __name__ == "__main__":
    onnx_path = './weights/FaceD_sim.onnx'
    engine_path = './weights/tmp.engine'
    # inputs = np.random.random((1, 3, 1024, 853)).astype(np.float32)
    engine = build_engine(onnx_path)
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
