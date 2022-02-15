all: ray.out

ray.out: sample_ray_tracing.cpp  tgaimage.cpp
	g++ sample_ray_tracing.cpp  tgaimage.cpp -o ray.out

clean: 
	rm ray.out

remove:
	rm  res/?* 