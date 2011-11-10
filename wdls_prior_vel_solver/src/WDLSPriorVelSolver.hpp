/*******************************************************************************
 *                 This file is part of the iTaSC project                      *
 *                        													   *
 *                        (C) 2011 Dominick Vanthienen                         *
 *                        (C) 2010 Ruben Smits                                 *
 *                        ruben.smits@mech.kuleuven.be                         *
 *                    dominick.vanthienen@mech.kuleuven.be,                    *
 *                    Department of Mechanical Engineering,                    *
 *                   Katholieke Universiteit Leuven, Belgium.                  *
 *                   http://www.orocos.org/itasc                               *
 *                                                                             *
 *       You may redistribute this software and/or modify it under either the  *
 *       terms of the GNU Lesser General Public License version 2.1 (LGPLv2.1  *
 *       <http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>) or (at your *
 *       discretion) of the Modified BSD License:                              *
 *       Redistribution and use in source and binary forms, with or without    *
 *       modification, are permitted provided that the following conditions    *
 *       are met:                                                              *
 *       1. Redistributions of source code must retain the above copyright     *
 *       notice, this list of conditions and the following disclaimer.         *
 *       2. Redistributions in binary form must reproduce the above copyright  *
 *       notice, this list of conditions and the following disclaimer in the   *
 *       documentation and/or other materials provided with the distribution.  *
 *       3. The name of the author may not be used to endorse or promote       *
 *       products derived from this software without specific prior written    *
 *       permission.                                                           *
 *       THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR  *
 *       IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED        *
 *       WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE    *
 *       ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,*
 *       INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    *
 *       (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS       *
 *       OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) *
 *       HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,   *
 *       STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING *
 *       IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE    *
 *       POSSIBILITY OF SUCH DAMAGE.                                           *
 *                                                                             *
 *******************************************************************************/

#ifndef _ITASC_WDLSVELOCITYSOLVER_HPP
#define _ITASC_WDLSVELOCITYSOLVER_HPP

#include <rtt/TaskContext.hpp>
#include <algorithm>
#include <itasc_core/Solver.hpp>
#include <itasc_core/choleski_semidefinite.hpp>
#include <Eigen/LU>

#include <vector>
#include <string>
#include <sstream>

namespace iTaSC {

class WDLSPriorVelSolver: public Solver {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	WDLSPriorVelSolver(const std::string& name);
	~WDLSPriorVelSolver() {};//destructor
	virtual bool configureHook();
	virtual bool startHook()
	{
		return true;
	};
	virtual void updateHook() {};
	virtual void stopHook() {};
	virtual void cleanupHook() {};
	virtual bool solve();
private:
	unsigned int nc_local, nq_local;

	/// attribute: number of priorities involved (default = 1)
	unsigned int priorityNo;
	/// property: Capacity to be allocated for the vectors containing the used weights. (default = 1000)
	unsigned int Wcapacity;
	/// property: precision of matrix comparison
	double precision;
	/// property: Maximum norm of (J#)Y with J# the WDLS-pseudoinverse of J. (default = 5.75 = out of maximum joint velocities for Kuka LWR robot)
	double bound_max;
	/// property: precision eigenvalue inverting
	double Seps;

	unsigned int testPRIOR;

	//helper variables for WDLS
	double dlambda;
	double lambda;
	//smalles eigen value of J
	double sigma_min;

	std::string externalName;
	std::stringstream ssName;

	//Weight on Jacobian in joint space= Wq
	Eigen::MatrixXd Wq;
	//inverse of Lq with Wq = Lq^T Lq
	Eigen::MatrixXd LqInv;
	// auxiliary variable to calculate LqInv
	Eigen::MatrixXd Lq;
	Eigen::MatrixXd LqT;
	Eigen::VectorXd qdot;
	Eigen::VectorXd qdot_prev;
	Eigen::VectorXd qdot_extra;

	Eigen::MatrixXd V, LqInv_V;
	//auxiliary matrices for SVD calculation
	Eigen::VectorXd tmp, S;
	//inverse of S: damped
	Eigen::MatrixXd SinvD;
	//inverse of S: NOT damped
	Eigen::MatrixXd Sinv;
	///projection matrix
	Eigen::MatrixXd P;
	///projection matrix of previous priority
	Eigen::MatrixXd P_prev;
	///part to be extracted from P for the next priority
	Eigen::MatrixXd P_extra;

	std::vector<int> nc_priorities;
	RTT::InputPort<std::vector<int> > nc_priorities_port;

	RTT::OutputPort<double> lambda_port; //TODO delete this
	RTT::OutputPort<Eigen::VectorXd> S_port; //TODO delete this

	//std::vector priorities contains structs of the "Priority" type
	struct Priority
	{
	public:
		unsigned int nc_priority;
		RTT::InputPort<Eigen::MatrixXd> A_port;
		RTT::InputPort<Eigen::MatrixXd> Wy_port;
		RTT::InputPort<Eigen::VectorXd> ydot_port;

		//generalized jacobian for a subtask with a certain priority
		Eigen::MatrixXd A_priority;
		//A projected on the null space of previous priorities
		Eigen::MatrixXd A_p;
		//WDLS pseudo-inverse of A_projected
		Eigen::MatrixXd ApInv_WDLS;
		//WLS pseudo-inverse of A_projected
		Eigen::MatrixXd ApInv_WLS;
		//weight in the task space for the generalized jacobian = Wy = Ly^T Ly
		Eigen::MatrixXd Wy_priority;
		//weight in the task space for the generalized jacobian = Wy = Ly^T Ly
		Eigen::MatrixXd Ly_priority;
		//inverse of Ly_priority
		Eigen::MatrixXd Lyinv_priority;
		//Ly^T = LyT
		Eigen::MatrixXd LyT;
		//task space coordinates
		Eigen::VectorXd ydot_priority;
		//compensation (for part of solution already met in lower priorities) term for ydot
		Eigen::VectorXd ydot_ct;
		//compensated ydot: compensated with the compensation term ydot_ct
		Eigen::VectorXd ydot_comp;
		//A*P*LqInv
		Eigen::MatrixXd Ap_LqInv;
		//Ly*A*P*LqInv
		Eigen::MatrixXd Ly_Ap_LqInv;
		//matrices for the SVD decomposition: U
		Eigen::MatrixXd U;

		//U^T*Ly
		Eigen::MatrixXd Ut_Ly;
		//Sinv*U^T*Ly
		Eigen::MatrixXd Sinv_Ut_Ly;

		Priority() :
			//initialisations
			nc_priority(0)
		{

		}
	};
	std::vector<Priority*> priorities;

	//flag to see whether a certain weight is already inserted
	bool newEntry;
	unsigned int weightIndex;
	std::vector<Eigen::MatrixXd> Wvector;
	std::vector<Eigen::MatrixXd> Lvector;
	std::vector<Eigen::MatrixXd> LinvVector;

};//end class definition

}//end namespace
#endif
