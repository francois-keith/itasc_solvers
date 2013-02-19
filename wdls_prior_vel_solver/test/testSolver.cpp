#include "HQPVelSolver.hpp"
#include "WDLSPriorVelSolver.hpp"

#include <rtt/Logger.hpp>

struct Priority
{
public:
	unsigned int nc_;		// priority
	RTT::OutputPort<Eigen::MatrixXd> A_port_;   //  Jacobian
	RTT::OutputPort<Eigen::MatrixXd> Wy_port_;  //  weight
	RTT::OutputPort<Eigen::VectorXd> ydot_port_;//

	//generalized jacobian for a subtask with a certain priority
	Eigen::MatrixXd A_;

	//weight in the task space for the generalized jacobian = Wy = Ly^T Ly
	Eigen::MatrixXd Wy_;

	//task space coordinates
	Eigen::VectorXd ydot_;

	void display () const
	{
	}

	Priority(int nc) :
		//initialisations
		nc_(nc)
	{
	}
};

int ORO_main(int arc, char* argv[])
{
	using namespace iTaSC;

	// Create the itasc solver
	Solver* sot = new HQPVelSolver("sot_solver");

	// Create the soth solver
	Solver* wld = new WDLSPriorVelSolver("itasc_solver");

	// Create the 'tasks'.
	// very simple case where two equality tasks are conflicting.
	// J1 = [[1, 0, 0], [0, 1, 0]], e1 = [1, 1]
	// J2 = [[0, 1, 0], [0, 0, 1]], e2 = [2, 2]
	std::vector<Priority*> taskVector;
	taskVector[0] = new Priority(0);
	taskVector[0]->A_.resize(3,2);
	taskVector[0]->A_ << 1, 0, 0,   0, 1, 0;
	taskVector[0]->ydot_.resize(2);
	taskVector[0]->ydot_ << 1, 1;

	taskVector[1] = new Priority(0);
	taskVector[1]->A_.resize(3,2);
	taskVector[1]->A_ << 0, 1, 0,   0, 0, 1;
	taskVector[1]->ydot_.resize(2);
	taskVector[1]->ydot_ << 2, 2;

	// publish Everything!
	for (unsigned i=0; i<taskVector.size(); ++i)
	{
		taskVector[i]->A_port_.write(taskVector[i]->A_);
		taskVector[i]->ydot_port_.write(taskVector[i]->ydot_);
//		sot->connectPorts(taskVector[i]->A_port_);
//		connectPorts(taskVector[i]->A_port_, sot->getPort(taskVector[i]->A_port_.getName()));
	}
//	RTT::InputPort<Eigen::MatrixXd> * wqsot = sot->getPort("Wq");
// Compute
//	sot->solve(); 
//	wld->solve();

	// Get and compare the results.
	RTT::InputPort<Eigen::VectorXd> qdot_soth_port;
	RTT::InputPort<Eigen::VectorXd> qdot_itasc_port;
	Eigen::VectorXd solution_soth;
	Eigen::VectorXd solution_wld;
//
	if(qdot_soth_port.read(solution_soth)== RTT::NoData)
		log(RTT::Error) << "No data on qdot_soth_port" << RTT::endlog();

	if(qdot_itasc_port.read(solution_wld)== RTT::NoData)
		log(RTT::Error) << "No data on qdot_itasc_port" << RTT::endlog();

	std::cout << "soth " << solution_soth.transpose() << std::endl;
	std::cout << "wld  " << solution_wld.transpose() << std::endl;
	// 

	// Clean up
	delete wld;
	delete sot;
	return 0;
}

