#include "HQPVelSolver.hpp"

#include <ocl/Component.hpp>

#include <kdl/utilities/svd_eigen_HH.hpp>
#include <rtt/Logger.hpp>

#include <rtt/os/TimeService.hpp>

ORO_CREATE_COMPONENT( iTaSC::HQPVelSolver );

namespace iTaSC {
	using namespace Eigen;
	using namespace std;
	using namespace KDL;
	using namespace RTT;

	HQPVelSolver::HQPVelSolver(const string& name) :
		Solver(name, false),
		hsolver(),
		Ctasks(),
		btasks(),
		solution()
	{
		//nq is already an attribute due to Solver.hpp
		this->addPort("nc_priorities",nc_priorities_port).doc("Port with vector of number of constraints per priority.");
		this->addPort("damping",damping_port).doc("Port with damping factor for the solver.");

		this->provides()->addAttribute("priorityNo_", priorityNo_); //"Number of priorities involved. (default = 1)"
		this->provides()->addProperty("precision", precision).doc("Precision for the comparison of matrices. (default = 1e-3)");

		//solve is already added as an operation in Solver.hpp
	}

	HQPVelSolver::~HQPVelSolver()
	{
		for (unsigned int i=0;i<priorityNo_;i++)
			delete priorities[i];
	}


	// Resize the problem
	bool HQPVelSolver::configureHook()
	{
		Logger::In in(this->getName());
		// the number of priority corresponds to the number of tasks
		priorities.resize(priorityNo_);
		nc_priorities_port.read(nc_priorities);

		for (unsigned int i=0;i<priorityNo_;i++)
		{
			priorities[i] = new Priority();
			priorities[i]->nc_priority = nc_priorities[i];
			log(Info) << " [Configuring] Got following number of constraints for priority "<< i+1 << " = \n " << nc_priorities[i] << endlog();

			//create attributes
			ssName.clear();
			ssName << "nc_" << i+1;
			ssName >> externalName;
			this->provides()->addAttribute(externalName, priorities[i]->nc_priority);//"Number of constraint coordinates for the priority of the index."
			//this->provides()->setValue( new Attribute<unsigned int>(externalName,priorities[i]->nc_priority) );
		}

		qdot.resize(nq);
		solution.resize( nq );

		//priority dependent creations/ initializations
		for (unsigned int i=0;i<priorityNo_;i++)
		{
			//create ports
			ssName.clear();
			ssName << "A_" << i+1;
			ssName >> externalName;
			this->ports()->addPort(externalName, priorities[i]->A_port).doc("Generalized Jacobian with priority of the index");
			ssName.clear();
			ssName << "Wy_" << i+1;
			ssName >> externalName;
			this->ports()->addPort(externalName, priorities[i]->Wy_port).doc("Output weight matrix with priority of the index");
			ssName.clear();
			ssName << "ydot_" << i+1;
			ssName >> externalName;
			this->ports()->addPort(externalName, priorities[i]->ydot_port).doc("Desired output velocity with priority of the index");

			//initializations
			priorities[i]->A_priority.resize(priorities[i]->nc_priority, nq);
			priorities[i]->A_priority.setZero();
			priorities[i]->Wy_priority.resize(priorities[i]->nc_priority, priorities[i]->nc_priority);
			priorities[i]->Wy_priority.setIdentity();
			priorities[i]->ydot_priority.resize(priorities[i]->nc_priority);
			priorities[i]->ydot_priority.setZero();
		}

		return true;
	}

	bool HQPVelSolver::startHook()
	{
		return true;
	}
	void HQPVelSolver::updateHook() {}
	void HQPVelSolver::stopHook() {}
	void HQPVelSolver::cleanupHook() {}


	/* Return true iff the solver sizes fit to the task set. */
	bool HQPVelSolver::checkSolverSize( )
	{
		assert( nq>0 );

		if(! hsolver ) return false;
		if( priorityNo_ != hsolver->nbStages() ) return false;

		bool toBeResized=false;
		/* TODO
		for( unsigned i=0;i<priorityNo_;++i )
		{
			assert( Ctasks[i].cols() == nbDofs && Ctasks[i].rows() == btasks[i].size() );
			TaskAbstract & task = *stack[i];
			if( btasks[i].size() != (int)task.taskSOUT.accessCopy().size() )
			{
				toBeResized = true;
				break;
			}
		}
		*/

		return !toBeResized;
	}


	bool HQPVelSolver::solve()
	{
		Logger::In in(this->getName());
#ifndef NDEBUG
		// start ticking
		RTT::os::TimeService::ticks time_begin = os::TimeService::Instance()->getTicks();
#endif //NDEBUG

		//initialize (useful?)
		qdot.setZero();

		if(! checkSolverSize() ) resizeSolver();

		using namespace soth;
		double damping = 1e-2;
		if( damping_port.read(damping) != RTT::NoData)
		{
			/* Only damp the final stage of the stack, 'cose of the solver known limitation. */
			hsolver->setDamping( 0 );
			hsolver->useDamp( true );
			hsolver->stages.back()->damping( damping );
		}
		else
		{
			hsolver->useDamp( false );
		}

		//priority loop
		for (unsigned int i=0;i<priorityNo_;i++)
		{
			//TODO check  the weight of each level	is 1
			if(priorities[i]->Wy_port.read(priorities[i]->Wy_priority)== RTT::NoData)
			{
				//assert( true );
			}

			//rea

			// the input ports
			if(priorities[i]->A_port.read(priorities[i]->A_priority)== RTT::NoData)
				log(Error) << "No data on A_port" << endlog();

			if(priorities[i]->ydot_port.read(priorities[i]->ydot_priority)== RTT::NoData)
				log(Error) << "No data on ydot_port" << endlog();

			// Fill the solver.
			MatrixXd & Ctask = Ctasks[i];
			Ctask = priorities[i]->A_priority;

			VectorBound & btask = btasks[i];
			const int nx1 = priorities[i]->ydot_priority.size();
			for( int c=0;c<nx1;++c )
				btask[c] = priorities[i]->ydot_priority[c];
		}

		// compute the solution
		hsolver->reset();
		hsolver->setInitialActiveSet();
		hsolver->activeSearch(solution);

		// TODO: handle the free floating base properly.
		bool controlFreeFloating = true;
		if( controlFreeFloating )
		{
			qdot=solution;
		}
		else


		{
			qdot=solution.tail( nq-6 );
		}


		qdot_port.write(qdot);

	//	#ifndef NDEBUG
	//	log(Debug) <<"qdot written \n"<< qdot << endlog();
	//	log(Debug) << "It took " << os::TimeService::Instance()->secondsSince(time_begin) << " seconds to solve." << endlog();
	//	#endif

		return true;
	}


	/** Knowing the sizes of all the stages (except the task ones),
	 * the function resizes the matrix and vector of all stages
	 */
	void HQPVelSolver::resizeSolver( )
	{
		// warning: loss of memory possible?
		hsolver = hcod_ptr_t(new soth::HCOD( nq, priorityNo_ ));
		Ctasks.resize(priorityNo_);
		btasks.resize(priorityNo_);

		for(unsigned i=0; i<priorityNo_;++i)
		{
			if(priorities[i]->ydot_port.read(priorities[i]->ydot_priority)== RTT::NoData)
				log(Error) << "No data on ydot_port. Call by HQPVelSolver::resizeSolver" << endlog();

			const int nx = priorities[i]->ydot_priority.size();
			Ctasks[i].resize(nx,nq);
			btasks[i].resize(nx);

			hsolver->pushBackStage( Ctasks[i],btasks[i] );

			ssName.clear();
			ssName << "nc_" << i+1;
			ssName >> externalName;

			hsolver->stages.back()->name = externalName;
		}

		solution.resize( nq );
	}
}


