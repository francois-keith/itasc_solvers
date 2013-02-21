#ifndef _ITASC_HQP_SOLVER_HPP
#define _ITASC_HQP_SOLVER_HPP

#include <rtt/TaskContext.hpp>
#include <algorithm>
#include <itasc_core/Solver.hpp>
#include <itasc_core/choleski_semidefinite.hpp>
#include <soth/HCOD.hpp>
#include <Eigen/LU>

#include <vector>
#include <string>
#include <sstream>

namespace iTaSC {

class HQPSolver: public Solver {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	HQPSolver(const std::string& name);
	virtual ~HQPSolver();//destructor

	virtual bool configureHook();
	virtual bool startHook();
	virtual void updateHook();
	virtual void stopHook();
	virtual void cleanupHook();
	virtual bool solve();

	void resizeSolver( );
	bool checkSolverSize( );

private:
	// the solver
	typedef boost::shared_ptr<soth::HCOD> hcod_ptr_t;
	hcod_ptr_t hsolver;

	// Jacobian of the tasks
	std::vector< Eigen::MatrixXd > Ctasks;

	// Upper/ Lower bounds for each tasks
	std::vector< soth::VectorBound > btasks;

	// Solution found by the solver. Size: size(dq) or size(dq+6)
	Eigen::VectorXd solution;

	// ?
	Eigen::VectorXd qdot;


	// attribute: number of priorities involved (default = 1)
	unsigned int priorityNo;

	// property: Capacity to be allocated for the vectors containing the used weights. (default = 1000)
	// unsigned int Wcapacity;

	// property: precision of matrix comparison
	double precision;


	std::vector<int> nc_priorities;
	RTT::InputPort<std::vector<int> > nc_priorities_port;

	// Port for the damping value.
	RTT::InputPort< double > damping_port;


	//std::vector priorities contains structs of the "Priority" type
	struct Priority
	{
	public:
		unsigned int nc_priority;
		RTT::InputPort<Eigen::MatrixXd> A_port;
		RTT::InputPort<Eigen::MatrixXd> Wy_port;

		// port for the value of the error and the bound min.
		RTT::InputPort<Eigen::VectorXd> ydot_port;
		// port for the max bound.
		RTT::InputPort<Eigen::VectorXd> ydot_max_port;
		// port for the max bound.
		RTT::InputPort<Eigen::VectorXd> inequalities_port;

		//generalized jacobian for a subtask with a certain priority
		Eigen::MatrixXd A_priority; // TODO: rename ?

		//weight in the task space for the generalized jacobian = Wy = Ly^T Ly
		Eigen::MatrixXd Wy_priority; // TODO: rename ?

		//task space coordinates
		Eigen::VectorXd ydot_priority; // TODO: rename ?
		Eigen::VectorXd ydot_priority_max; // TODO: rename ?

		Eigen::VectorXd inequalities; // TODO: rename ?

		Priority() :
			//initializations
			nc_priority(0)
		{

		}
	};
	std::vector<Priority*> priorities;

// temporary data only used to avoid time consuption at the creation.
private:
	std::string externalName;
	std::stringstream ssName;
};//end class definition

}//end namespace
#endif
