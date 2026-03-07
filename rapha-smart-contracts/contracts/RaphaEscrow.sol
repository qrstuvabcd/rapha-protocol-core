// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

contract RaphaEscrow {
    address public protocolTreasury;
    
    struct Job {
        address funder;
        uint256 amount;
        bool isSettled;
    }
    
    mapping(string => Job) public jobs;
    
    event JobFunded(string jobId, address funder, uint256 amount);
    event JobSettled(string jobId, address nodeAddress, uint256 nodePayout, uint256 protocolFee);
    
    constructor(address _protocolTreasury) {
        protocolTreasury = _protocolTreasury;
    }
    
    // In MVP, we mock the ERC20 by just using native currency
    function fundJob(string memory jobId) external payable {
        require(msg.value > 0, "Must fund with some value");
        require(jobs[jobId].amount == 0, "Job already funded");
        
        jobs[jobId] = Job({
            funder: msg.sender,
            amount: msg.value,
            isSettled: false
        });
        
        emit JobFunded(jobId, msg.sender, msg.value);
    }
    
    function verifyAndPay(string memory jobId, bool validProof, address payable nodeAddress) external {
        Job storage job = jobs[jobId];
        require(job.amount > 0, "Job not found");
        require(!job.isSettled, "Job already settled");
        require(validProof, "Invalid ZK proof");
        
        job.isSettled = true;
        
        uint256 totalAmount = job.amount;
        uint256 protocolFee = (totalAmount * 5) / 100;
        uint256 nodePayout = totalAmount - protocolFee;
        
        // Payout to Node and Protocol Treasury
        nodeAddress.transfer(nodePayout);
        payable(protocolTreasury).transfer(protocolFee);
        
        emit JobSettled(jobId, nodeAddress, nodePayout, protocolFee);
    }
}
