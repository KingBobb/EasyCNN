#pragma once

#include"Configure.h"
#include"Layer.h"

#include<memory>


namespace EasyCNN
{
	class NetWork
	{
	public:
		NetWork();
		virtual~NetWork();
		void addlayer(std::shared_ptr<Layer> layer);
		void backward();
		void forward();
	private:

	};
}