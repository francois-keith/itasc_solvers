#include "HQPSolver.hpp"

#include <ocl/Component.hpp>

#include <kdl/utilities/svd_eigen_HH.hpp>
#include <rtt/Logger.hpp>

#include <rtt/os/TimeService.hpp>

ORO_CREATE_COMPONENT( iTaSC::HQPSolver );

namespace
{
	// A tool method. To be removed, hopefully.
	bool isInVector(unsigned c, const std::vector<unsigned> & v)
	{
		for (unsigned i=0; i < v.size(); ++i)
			if (v[i] == c)
				return true;
		return false;
	}
}

namespace iTaSC {
	using namespace Eigen;
	using namespace std;
	using namespace KDL;
	using namespace RTT;

	HQPSolver::HQPSolver(const string& name) :
		Solver(name, true),
		hsolver(),
		Ctasks(),
		btasks(),
		solution()
	{
		//nq is already an attribute due to Solver.hpp
		this->addPort("nc_priorities",nc_priorities_port).doc("Port with vector of number of constraints per priority.");
		this->addPort("damping",damping_port).doc("Port with damping factor for the solver.");

		this->provides()->addAttribute("priorityNo", priorityNo); //"Number of priorities involved. (default = 1)"
		this->provides()->addProperty("precision", precision).doc("Precision for the comparison of matrices. (default = 1e-3)");

		//solve is already added as an operation in Solver.hpp
	}

	HQPSolver::~HQPSolver()
	{
		for (unsigned int i=0;i<priorityNo;i++)
			delete priorities[i];
	}


	// Resize the problem
	bool HQPSolver::configureHook()
	{
		Logger::In in(this->getName());
		// the number of priority corresponds to the number of tasks
		priorities.resize(priorityNo);
		nc_priorities_port.read(nc_priorities);

		qdot.resize(nq);
		solution.resize( nq );

		for (unsigned int i=0;i<priorityNo;i++)
		{
			priorities[i] = new Priority();
			priorities[i]->nc_priority = nc_priorities[i];
			log(Info) << " [Configuring] Got following number of constraints for priority "<< i+1 << " = \n " << nc_priorities[i] << endlog();

			//create attributes
			ssName.clear();
			ssName << "nc_" << i+1;
			ssName >> externalName;
			//"Number of constraint coordinates for the priority of the index."
			this->provides()->addAttribute(externalName, priorities[i]->nc_priority);

			//priority dependent creations/ initializations
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
			this->ports()->addPort(externalName, priorities[i]->ydot_port).doc(
						"Desired output velocity (or lower bound if ineq) with priority of the index");

			ssName.clear();
			ssName << "ydot_max_" << i+1;
			ssName >> externalName;
			this->ports()->addPort(externalName, priorities[i]->ydot_max_port).doc(
						"Upper bound for the desired output velocity with priority of the index");

			ssName.clear();
			ssName << "inequalities_" << i+1;
			ssName >> externalName;
			this->ports()->addPort(externalName, priorities[i]->inequalities_port).doc("Inequality flag: which tasks are the inequalities?");

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

	bool HQPSolver::startHook()
	{
		return true;
	}
	void HQPSolver::updateHook() {}
	void HQPSolver::stopHook() {}
	void HQPSolver::cleanupHook() {}


	/* Return true iff the solver sizes fit to the task set. */
	bool HQPSolver::checkSolverSize( )
	{
		assert( nq>0 );

		if(! hsolver ) return false;
		if( priorityNo != hsolver->nbStages() ) return false;

		bool toBeResized=false;
		for( unsigned i=0;i<priorityNo;++i )
		{
			assert( Ctasks[i].cols() == nq && Ctasks[i].rows() == priorities[i]->nc_priority );
			if( btasks[i].size() != priorities[i]->nc_priority)
			{
				toBeResized = true;
				break;
			}
		}

		return !toBeResized;
	}


	bool HQPSolver::solve()
	{
		Logger::In in(this->getName());
#ifndef NDEBUG
		// start ticking
		RTT::os::TimeService::ticks time_begin = os::TimeService::Instance()->getTicks();
#endif //NDEBUG

		// verifivation Wq == identity.
		Eigen::MatrixXd Wq;
		if( Wq_port.read(Wq) != RTT::NoData &&  Wq.isIdentity() == false)
		{
			log(Error) << " HQP solver only handles the case where Wq = Identity " << endlog();
			return false;
		}


		//initialize (useful?)
		qdot.setZero();

		if(! checkSolverSize() )
			resizeSolver();

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
		// for each task group
		for (unsigned int i=0;i<priorityNo;i++)
		{
			//TODO check that the tasks are not weighted (yet)
			if(priorities[i]->Wy_port.read(priorities[i]->Wy_priority) != RTT::NoData
				 && priorities[i]->Wy_priority.isIdentity() == false)
			{
				log(Error) << " HQP solver only handles the case where Wy = Identity " << endlog();
				return false;
			}

			// priorities[i]->inequalities lists the dof in equality and in equality
			if(priorities[i]->inequalities_port.read(priorities[i]->inequalities) == RTT::NoData)
				priorities[i]->inequalities.resize(0); // no inequalities.
			else if ( (priorities[i]->inequalities.size() != 0)
				&& (priorities[i]->inequalities.size() != priorities[i]->nc_priority))
					log(Error) << "Incorrect size for the vector inequalities." << endlog();


			// -- Handle the tasks.

			// the input ports
			if(priorities[i]->A_port.read(priorities[i]->A_priority)== RTT::NoData)
				log(Error) << "No data on A_port" << endlog();

			if(priorities[i]->ydot_port.read(priorities[i]->ydot_priority)== RTT::NoData)
				log(Error) << "No data on ydot_port" << endlog();

			// Fill the solver: the jacobian.
			MatrixXd & Ctask = Ctasks[i];
			Ctask = priorities[i]->A_priority;

			// Fill the solver: the reference.
			VectorBound & btask = btasks[i];
			const unsigned nx1 = priorities[i]->ydot_priority.size();

			//equality task.
			if(priorities[i]->inequalities.size() == 0)
			{
				for( unsigned c=0;c<nx1;++c )
					btask[c] = priorities[i]->ydot_priority[c];
			}
			else
			{
				// only the lower bound is given. Correct only in the case where the
				//  constraints are equality tasks or lb inequality tasks
				if(priorities[i]->ydot_max_port.read(priorities[i]->ydot_priority_max)== RTT::NoData)
				{
					bool upperBoundUseless = true;
					for( unsigned c=0;c<nx1;++c )
						if ( (priorities[i]->inequalities[c] != 0) && (priorities[i]->inequalities[c] != 1))
							upperBoundUseless = false;
					if(upperBoundUseless == false)
					{
						log(Error) << "No data on ydot_max_port" << endlog();
						log(Error) << "For now, we only handle double inequalities." << endlog();
						return false;
					}
				}

				for( unsigned c=0;c<nx1;++c )
				{
					// if this is a unilateral constraint
					switch ( priorities[i]->inequalities[c] )
					{
						case(0):
							btask[c] = priorities[i]->ydot_priority[c];
							break;
						case(1):
							btask[c] = Bound( priorities[i]->ydot_priority[c], Bound::BOUND_INF);
							break;
						case(2):
							btask[c] = Bound( priorities[i]->ydot_priority_max[c], Bound::BOUND_SUP);
							break;
						case(3):
							assert(ydot_priority_max.size() == ydot_priority.size());
							assert(priorities[i]->ydot_priority[c] <= priorities[i]->ydot_priority_max[c]);
							btask[c] = std::pair<double,double>(
								priorities[i]->ydot_priority[c],
								priorities[i]->ydot_priority_max[c]
							);
							break;
						default:
							std::cerr << " Unknown inequality type: " << priorities[i]->inequalities[c] << std::endl;
						}
				}
			}
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

		#ifndef NDEBUG
		log(Debug) <<"qdot written \n"<< qdot << endlog();
		log(Debug) << "It took " << os::TimeService::Instance()->secondsSince(time_begin) << " seconds to solve." << endlog();
		#endif

		return true;
	}


	/** Knowing the sizes of all the stages (except the task ones),
	 * the function resizes the matrix and vector of all stages
	 */
	void HQPSolver::resizeSolver( )
	{
		// warning: loss of memory possible?
		hsolver = hcod_ptr_t(new soth::HCOD( nq, priorityNo ));
		Ctasks.resize(priorityNo);
		btasks.resize(priorityNo);

		for(unsigned i=0; i<priorityNo;++i)
		{
			if(priorities[i]->ydot_port.read(priorities[i]->ydot_priority)== RTT::NoData)
				log(Error) << "No data on ydot_port. Called by HQPSolver::resizeSolver" << endlog();

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


