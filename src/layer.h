#include "vulkan_utils.h"
#include <queue>
#include <functional>

struct CallbackQueue
{
	std::deque<std::function<void()>> functions;

	void push_function(std::function<void()>&& function) {
		functions.push_back(function);
	}

	void flush() {
        // executes function FIFO
		for (auto it = functions.begin(); it != functions.end(); it++) {
			(*it)();
		}
	}
};
struct CallbackStack
{
	std::deque<std::function<void()>> functions;

	void push_function(std::function<void()>&& function) {
		functions.push_back(function);
	}

	void flush() {
        // executes function FILO
		for (auto it = functions.rbegin(); it != functions.rend(); it++) {
			(*it)();
		}
	}
};

class Layer
{
private:
    std::vector<vk::Semaphore> _waitSemaphores = {};
    std::vector<vk::PipelineStageFlags> _waitStages = {};
    std::vector<vk::Semaphore> _signalSemaphores = {};

    CallbackQueue onSwapChainDelete; // handles deletion callbacks on resize
    CallbackStack onSwapChainRecreate; // handles recreation callbacks on resize
    
public:
    Layer();
    ~Layer();

	void init();
    void await(vk::Semaphore semaphore, vk::PipelineStageFlags stages); // Set on which semaphore which stage should wait
    void signal(vk::Semaphore semaphore); // Set which semaphores should be signaled
    void onUpdate(float dt); // wait for semaphores, do update, submit and signal semaphore

};
