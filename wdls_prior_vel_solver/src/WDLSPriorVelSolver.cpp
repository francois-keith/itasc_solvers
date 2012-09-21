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
#include "WDLSPriorVelSolver.hpp"

#include <ocl/Component.hpp>

#include <kdl/utilities/svd_eigen_HH.hpp>
#include <rtt/Logger.hpp>

#include <rtt/os/TimeService.hpp>
#include <Eigen/LU>
ORO_CREATE_COMPONENT( iTaSC::WDLSPriorVelSolver );

namespace iTaSC {
using namespace Eigen;
using namespace std;
using namespace KDL;
using namespace RTT;

WDLSPriorVelSolver::WDLSPriorVelSolver(const string& name) :
	Solver(name, 0),
	nc_local(0),
	nq_local(0),
	priorityNo(1),
	Wcapacity(1000),
	precision(1e-3),
	bound_max(5.75),
	Seps(1e-9),
	nc_priorities(std::vector<int>(1,0))
{	
	this->addPort("nc_priorities",nc_priorities_port).doc("Port with vector of number of constraints per priority.");
	this->addPort("lambda_port", lambda_port).doc("delete me");
	this->addPort("S_port", S_port).doc("delete me");
	//nq is already an attribute due to Solver.hpp
	this->provides()->addAttribute("priorityNo", priorityNo);//"Number of priorities involved. (default = 1)"
	this->provides()->addProperty("Wcapacity", Wcapacity).doc("Capacity to be allocated for the vectors containing the used weights. (default = 1000)");
	this->provides()->addProperty("precision", precision).doc("Precision for the comparison of matrices. (default = 1e-3)");
	this->provides()->addProperty("bound_max", bound_max).doc("Maximum norm of (J#)Y with J# the WDLS-pseudoinverse of J. (default = 5.75)");
	this->provides()->addProperty("Seps", Seps).doc("precision eigenvalue inverting");
	//solve is already added as an operation in Solver.hpp
}

bool WDLSPriorVelSolver::configureHook() {
	Logger::In in(this->getName());
	priorities.resize(priorityNo) ;
	nc_priorities_port.read(nc_priorities);
#ifndef NDEBUG
	log(Debug) << "priorityNo = "<< priorityNo <<  endlog();
#endif
	for (unsigned int i=0;i<priorityNo;i++)
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
	nq_local = nq;
	Wvector.reserve(Wcapacity);
	Lvector.reserve(Wcapacity);
	LinvVector.reserve(Wcapacity);
	Wq.resize(nq_local, nq_local);
	LqInv.resize(nq_local, nq_local);
	LqT.resize(nq_local, nq_local);
	qdot.resize(nq_local);
	qdot_prev.resize(nq_local);
	qdot_extra.resize(nq_local);
	V.resize(nq_local, nq_local);
	V.setIdentity();
	LqInv_V.resize(nq_local, nq_local);
	S.resize( nq_local);
	S.setConstant(1.0);
	S_port.write(S);//TODO delete me!
	SinvD.resize( nq_local, nq_local);
	SinvD.setZero();
	Sinv.resize( nq_local, nq_local);
	Sinv.setZero();
	tmp.resize(nq_local);
	tmp.setZero();
	P.resize(nq_local,nq_local);
	P.setIdentity();
	P_prev.resize(nq_local,nq_local);
	P_prev.setIdentity();

	//priority dependent creations/ initializations
	for (unsigned int i=0;i<priorityNo;i++)
	{
		newEntry = true;
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
		priorities[i]->A_priority.resize(priorities[i]->nc_priority, nq_local);
		priorities[i]->A_priority.setZero();
		priorities[i]->A_p.resize(priorities[i]->nc_priority, nq_local);
		priorities[i]->A_p.setZero();
		priorities[i]->ApInv_WDLS.resize(nq_local, priorities[i]->nc_priority);
		priorities[i]->ApInv_WDLS.setZero();
		priorities[i]->ApInv_WLS.resize(nq_local, priorities[i]->nc_priority);
		priorities[i]->ApInv_WLS.setZero();
		priorities[i]->Wy_priority.resize(priorities[i]->nc_priority, priorities[i]->nc_priority);
		priorities[i]->Wy_priority.setIdentity();
		priorities[i]->ydot_priority.resize(priorities[i]->nc_priority);
		priorities[i]->ydot_priority.setZero();
		priorities[i]->ydot_ct.resize(priorities[i]->nc_priority);
		priorities[i]->ydot_ct.setZero();
		priorities[i]->ydot_comp.resize(priorities[i]->nc_priority);
		priorities[i]->ydot_comp.setZero();
		priorities[i]->Ap_LqInv.resize(priorities[i]->nc_priority, nq_local);
		priorities[i]->Ap_LqInv.setZero();
		priorities[i]->Ly_Ap_LqInv.resize(priorities[i]->nc_priority, nq_local);
		priorities[i]->Ly_Ap_LqInv.setZero();
		priorities[i]->U.resize(priorities[i]->nc_priority, nq_local);
		priorities[i]->U.setZero();
		priorities[i]->Ut_Ly.resize(nq_local, priorities[i]->nc_priority);
		priorities[i]->Ut_Ly.setZero();
		priorities[i]->Sinv_Ut_Ly.resize(nq_local, priorities[i]->nc_priority);
		priorities[i]->Sinv_Ut_Ly.setZero();
		//Put basic weights Wy in the W/L/Linv vector
		// basic weight: Wy = Identity matrix => Ly = Identity matrix
		for(unsigned int j=0; j<i; j++)
		{
			if(priorities[j]->nc_priority==priorities[i]->nc_priority)
			{
				newEntry = false;
			}
		}
		if(newEntry)
		{
			Wvector.insert(Wvector.end(), MatrixXd::Zero(priorities[i]->nc_priority, priorities[i]->nc_priority));
			Lvector.insert(Lvector.end(), MatrixXd::Zero(priorities[i]->nc_priority, priorities[i]->nc_priority));
			LinvVector.insert(LinvVector.end(), MatrixXd::Zero(priorities[i]->nc_priority, priorities[i]->nc_priority));

			Wvector.insert(Wvector.end(), MatrixXd::Identity(priorities[i]->nc_priority, priorities[i]->nc_priority));
			Lvector.insert(Lvector.end(), MatrixXd::Identity(priorities[i]->nc_priority, priorities[i]->nc_priority));
			LinvVector.insert(LinvVector.end(), MatrixXd::Identity(priorities[i]->nc_priority, priorities[i]->nc_priority));
		}
	}
	//Put basic weights Wq in the W/L/Linv vector
	// basic weight: Wq = Identity matrix => LqInv = Identity matrix
	newEntry = true;
	for(unsigned int k=0; k<priorities.size(); k++)
	{
		if(priorities[k]->nc_priority==nq_local)
		{
			newEntry = false;
		}
	}
	if(newEntry)
	{
		Wvector.insert(Wvector.end(), MatrixXd::Identity(nq_local,nq_local));
		Lvector.insert(Lvector.end(), MatrixXd::Identity(nq_local,nq_local));
		LinvVector.insert(LinvVector.end(), MatrixXd::Identity(nq_local,nq_local));
	}

	return true;
}

bool WDLSPriorVelSolver::solve() {
	Logger::In in(this->getName());
RTT::os::TimeService::ticks time_begin = os::TimeService::Instance()->getTicks();
	//initialization
	qdot.setZero();
	P.setIdentity(); //important initialization, do not change this value!

	//read the general ports
	Wq_port.read(Wq);

	//Check of this weight is already in the Wvector
	newEntry = true;
	for(unsigned int k=0; k<Wvector.size(); k++)
	{
		// check first whether the matrices have the same size (weighting matrix should be square: check only columns)
		if(Wq.cols()==Wvector[k].cols())
		{
			// check whether the matrix is the same
			if(Wq.isApprox(Wvector[k], precision) )
			{
				weightIndex = k;
				newEntry = false;
				break; //found an entry that is the same, don't look further for it
			}
		}
	}
	if(newEntry)
	{
		//Wq is not in the weightmap: calculate L/Linv and put it in vector
		//Cholesky factorization: Wq = (Lq^T)(Lq^T)^T
		if(!cholesky_semidefinite(Wq, LqT))
		{
			log(Error) << "Cholesky factorization of Wq went wrong" << endlog();
			return false;
		}
		Lq = LqT.transpose();
		LqInv = Lq.inverse();//TODO this isn't realtime !!!
		//check whether there is space left in the vectors
		if(LinvVector.size()==Wcapacity)
		{
			log(Warning) << "W / L / Linv vectors are at maximum capacity. \n A non real-time operation will take place to delete the first half of the vector." << endlog();
			Lvector.erase(Lvector.begin(), Lvector.begin() + (int) Lvector.size()/2);
			LinvVector.erase(LinvVector.begin(), LinvVector.begin() + (int) LinvVector.size()/2);
			Wvector.erase(Wvector.begin(), Wvector.begin() + (int) Wvector.size()/2);
			if(Wvector.size()!=Lvector.size() && Wvector.size()!=LinvVector.size())
			{
				log(Warning) << "W/L/Linv don't have the same size, clearing all three vectors" <<endlog();
				Wvector.clear();
				Lvector.clear();
				LinvVector.clear();
			}
		}
		//put Wq in the Wvector and the result in the LinvVector
		LinvVector.insert(LinvVector.end(), LqInv);
		//Wvector[LinvVector.size()-1] = Wq; //make sure that Wq is inserted at same index as corresponding LqInv
		//Lvector[LinvVector.size()-1] = Lq;
		// memory is reserved, so the following should be real-time? but, not completely sure that same index
		// eg if someone else inserts something in the vector Wvector or Lvector, but in normal situations this could not happen?
		// cause these two vector are _always_ accessed together
		Wvector.push_back(Wq);
		Lvector.push_back(Lq);

	}else
	{
		//Wq is in the Wvector: get it's LqInv
#ifndef NDEBUG
		log(Debug) << "LqInv found in the weightvector" <<  endlog();
#endif
		LqInv = LinvVector[weightIndex];
		if(LqInv.isZero()) //NOTE if problems: possibility to give eps with it, eg. isZero(1e-8)
		{
#ifndef NDEBUG
			log(Debug) << "Calculating LqInv from Lq"<< endlog();
#endif
			//LqInv hasn't been calculated yet
			LqInv = Lvector[weightIndex].inverse();//TODO this isn't realtime !!!
		}
	}
#ifndef NDEBUG
	log(Debug) << "Lq = \n" << Lq << endlog();
	log(Debug) << "LqInv = \n" << LqInv << endlog();
#endif


	//priority loop
#ifndef NDEBUG
	log(Debug)<<"priorityNo = "<<priorityNo << endlog();
#endif
	for (unsigned int i=0;i<priorityNo;i++)
	{
		//initialization
#ifndef NDEBUG
		log(Debug)<<"priorities[i]->nc_priority = "<< priorities[i]->nc_priority<< endlog();
		log(Debug) << "ydot_ct_"<<i<<" = " <<priorities[i]->ydot_ct << endlog();
#endif
		priorities[i]->ydot_ct.setZero();
		priorities[i]->ydot_comp.setZero();
		qdot_prev = qdot;
		P_prev = P;
		//read the input ports
		if(priorities[i]->A_port.read(priorities[i]->A_priority)== RTT::NoData)
		{
			log(Error) << "No data on A_port" << endlog();
		}
		if(priorities[i]->Wy_port.read(priorities[i]->Wy_priority)== RTT::NoData)
		{
			log(Error) << "No data on Wy_port" << endlog();
		}
		if(priorities[i]->ydot_port.read(priorities[i]->ydot_priority)== RTT::NoData)
		{
			log(Error) << "No data on ydot_port" << endlog();
		}

		//Check of this weight is already in the WyVector
		newEntry = true;
		for(unsigned int m=0; m<Wvector.size(); m++)
		{
			// check first whether the matrices have the same size (weighting matrix should be square: check only columns)
			if(priorities[i]->Wy_priority.cols()==Wvector[m].cols())
			{
				if( priorities[i]->Wy_priority.isApprox(Wvector[m], precision))
				{
					weightIndex = m;
					newEntry = false;

#ifndef NDEBUG
					log(Debug) << "the following two are considered the same" << endlog();
					log(Debug) << "priorities[i]->Wy_priority = " << priorities[i]->Wy_priority << endlog();
					log(Debug) << "Wvector[m] = " << Wvector[m] << endlog();
#endif

					break; //found an entry that is the same, don't look further for it
				}
			}
#ifndef NDEBUG
			log(Debug) << "Wvector content nr" << m << "= \n" << Wvector[m] << endlog();
#endif
		}
		if(newEntry)
		{
			//Wy is not in the weightmap: calculate Ly and put it in Lvector
			//Cholesky factorization: Wy = (Ly^T)(Ly^T)^T
			if(!cholesky_semidefinite(priorities[i]->Wy_priority, priorities[i]->LyT))
			{
				log(Error) << "Cholesky factorization of Wy went wrong" << endlog();
				return false;
			}
			priorities[i]->Ly_priority = priorities[i]->LyT.transpose();
			//check whether there is space left in the map
			if(Lvector.size()==Wcapacity)
			{
				log(Warning) << "W / L / Linv vectors are at maximum capacity. \n A non real-time operation will take place to delete the first half of the vector." << endlog();
				Lvector.erase(Lvector.begin(), Lvector.begin() + (int) Lvector.size()/2);
				LinvVector.erase(LinvVector.begin(), LinvVector.begin() + (int) LinvVector.size()/2);
				Wvector.erase(Wvector.begin(), Wvector.begin() + (int) Wvector.size()/2);
				if(Wvector.size()!=Lvector.size() && Wvector.size()!=LinvVector.size())
				{
					log(Warning) << "W/L/Linv don't have the same size, clearing all three vectors" <<endlog();
					Wvector.clear();
					Lvector.clear();
					LinvVector.clear();
				}
			}

			//put Wy in the WyVector and the result in the LyVector
			Lvector.push_back(priorities[i]->Ly_priority);
			Wvector.push_back(priorities[i]->Wy_priority);
			//calculating the inverse takes to much time, and it's not sure whether we'll use the result=> put a zero matrix and let the LqInv loop calculate the inverse himself out of L if needed
			LinvVector.push_back(MatrixXd::Zero(priorities[i]->Ly_priority.rows(), priorities[i]->Ly_priority.cols()));
			//make sure that Wq and Lqinv is inserted at same index as corresponding Lq
			if(!(Lvector.size() && Wvector.size() && LinvVector.size()))
			{
				log(Error) << "Mix up of weights!" << endlog();
				return false;
			}
		}else
		{
#ifndef NDEBUG
			log(Debug) << "Ly found in the weightvector"<< endlog();
#endif
			//Wy is in the WyVector: get it's Ly
			priorities[i]->Ly_priority = Wvector[weightIndex];
		}

		//print the values of the variables
#ifndef NDEBUG
		log(Debug) << "priorities[i]->A_priority = \n" << priorities[i]->A_priority << endlog();
		log(Debug) << "P = \n" << P << endlog();
		log(Debug) << "priorities[i]->Ly_priority = \n" << priorities[i]->Ly_priority << endlog();
		//log(Debug) << "qdot = \n" << qdot << endlog();
		log(Debug) << "priorities[i]->ydot_priority = \n" << priorities[i]->ydot_priority << endlog();
#endif

		//THE ACTUAL ALGORITHM
		//projection of A on the null space of previous priorities
		(priorities[i]->A_p).noalias() = (priorities[i]->A_priority*P);
		//calculation of weighted jacobian
		(priorities[i]->Ap_LqInv).noalias() = (priorities[i]->A_p * LqInv);
		(priorities[i]->Ly_Ap_LqInv).noalias() = (priorities[i]->Ly_priority * priorities[i]->Ap_LqInv);
		//compensation for part of solution already met in lower priorities
		(priorities[i]->ydot_ct).noalias() = (priorities[i]->A_priority*qdot);
		priorities[i]->ydot_comp = priorities[i]->ydot_priority - priorities[i]->ydot_ct;

		//SVD calculation of A_projected with weighting
#ifndef NDEBUG
			log(Debug) <<"Ly_Ap_LqInv \n"<< priorities[i]->Ly_Ap_LqInv << endlog();
		//	log(Debug) << "U \n" << priorities[i]->U << endlog();
		//	log(Debug) << "S \n" << priorities[i]->S << endlog();
		//	log(Debug) << "V \n" <<  V << endlog();
		//	log(Debug) << "tmp \n" << tmp << endlog();*/
#endif
		int ret = svd_eigen_HH(priorities[i]->Ly_Ap_LqInv, priorities[i]->U, S, V, tmp);
		if (ret < 0) {
			log(Error)<<"svd_eigen_HH on Ly_Ap_LqInv went fatal"<<endlog();
			this->fatal();
			return false;
		}

		//LqInv*V
		LqInv_V.noalias() = (LqInv * V);
		// use varying lambda, not fixed: approximation of an optimal lambda as proposed by:
		// the PhD thesis of P. Baerlocher, EPFL, Lausanne, 2001 (thesis no. 2383) based on:
		//A.A. Maciejewski, C.A. Klein, “Numerical Filtering for the Operation of
		//Robotic Manipulators through Kinematically Singular Configurations”,
		//Journal of Robotic Systems, Vol. 5, No. 6, pp. 527 - 552, 1988.

		//This way of calculating the SVD has a S matrix that has a size of nq x nq, possibly introducing extra 'zero' eigenvalues
		//(In theory, the S matrix should be a nc x nq matrix. When nc < nq, this extra zero is introduced, as is mostly the case with redundant robots and 6 constraints on a task.)
		sigma_min = S.block(0,0,min(nq_local, priorities[i]->nc_priority),1).minCoeff();
#ifndef NDEBUG
		// log(Debug) << "sigma_min = " << sigma_min << endlog();
#endif
		//dlambda = priorities[i]->ydot_priority.norm()/bound_max;//=as proposed by Baerlocher, this gives initialisation problems where ydot_priority = 0
		dlambda = 1/bound_max; //=as proposed by Maciejewski
		if(sigma_min <= dlambda/2)
		{
			lambda = dlambda/2;
		}else if(sigma_min >= dlambda){
			lambda =0;
		}else{
			lambda = sqrt(sigma_min*(dlambda-sigma_min));
		}
		lambda_port.write(lambda);
		S_port.write(S);
#ifndef NDEBUG
		log(Debug) << "lambda = " << lambda << endlog();
#endif
		//SinvD
		SinvD.setZero();
		for (unsigned int l = 0; l < min(nq_local, priorities[i]->nc_priority); l++)
		{
			SinvD(l, l) = (S(l) / (S(l) * S(l) + lambda * lambda));
		}
		//Sinv
		for (unsigned int l = 0; l < (unsigned int) S.rows(); l++)
		{
			if(S(l)<Seps) //this is KDL::epsilon of the KDL utilities
			{
				Sinv(l, l) = 0;
			}else
			{
				Sinv(l, l) = 1 / S(l);
			}
		}

		//U^T *Ly
		(priorities[i]->Ut_Ly).noalias() = (priorities[i]->U.transpose()*priorities[i]->Ly_priority);
		//SinvD*U^T*Ly
		(priorities[i]->Sinv_Ut_Ly).noalias() = (SinvD*priorities[i]->Ut_Ly);
		//WDLS pseudo-inverse of Aprojected = LqInv*(V*SinvD*U^T)*Ly = LqInv_V*SinvD*Ut_Ly = LqInv_V*Sinv_Ut_Ly
		(priorities[i]->ApInv_WDLS).noalias() = (LqInv_V*priorities[i]->Sinv_Ut_Ly);

		//qdot=LqInv*V*S^-1*U'*Ly'*ydot
		qdot_extra.noalias() = (priorities[i]->ApInv_WDLS*priorities[i]->ydot_comp);
#ifndef NDEBUG
		// log(Debug) << "qdot_extra = " << qdot_extra << endlog();
#endif
		//Extension of the solution in joint space with the implications of a task of the current priority
		qdot.noalias() = qdot_prev + qdot_extra;

		//Calculate projection matrix for the next priority
		//Sinv*U^T*Ly
		(priorities[i]->Sinv_Ut_Ly).noalias() = (Sinv*priorities[i]->Ut_Ly);
		//WLS pseudo-inverse of Aprojected = LqInv*(V*Sinv*U^T)*Ly = LqInv_V*Sinv*Ut_Ly = LqInv_V*Sinv_Ut_Ly
		(priorities[i]->ApInv_WLS).noalias() = (LqInv_V*priorities[i]->Sinv_Ut_Ly);
		(P_extra).noalias() = (priorities[i]->ApInv_WLS*priorities[i]->A_p);
		//P_k+1=P_k-pinv(Ap_k)*Ap_k
		P = P_prev - P_extra;
	}
	qdot_port.write(qdot);
#ifndef NDEBUG
	log(Debug) <<"qdot written \n"<< qdot << endlog();
	log(Debug) << "It took " << os::TimeService::Instance()->secondsSince(time_begin) << " seconds to solve." << endlog();
#endif
	return true;
}

}

