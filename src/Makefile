CXX = g++
CXXFLAGS = -I. -I../common/ -std=c++14 -O3
DEPS = 
OBJ = main.o simple_4arith_mod.o simple_mod.o

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

app: $(OBJ)
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean:
	rm -f *.o app
