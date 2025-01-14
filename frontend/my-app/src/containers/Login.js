import React from "react";
import { Form, Input, Button, Spin, Icon } from "antd";
import { UserOutlined, LockOutlined, LoadingOutlined } from "@ant-design/icons";
import { connect } from "react-redux";
import { NavLink } from "react-router-dom";
import * as actions from '../store/actions/auth';


const antIcon = <LoadingOutlined style={{ fontSize: 24 }} spin />;


const onFinishFailed = (errorInfo) => {
  console.log("Failed:", errorInfo);
};


class NormalLoginForm extends React.Component {
  handleSubmit = (values) => {
    console.log("Success:", values);
    
    
    this.props.onAuth(values.username, values.password);
      

    this.props.history.push('/');

  };


  render() {
    let errorMessage = null;
    if (this.props.error) {
      errorMessage = <p>{this.props.error.message}</p>;
    }


    return (
      <div>
        {errorMessage}
        {this.props.loading ? (
          <Spin indicator={antIcon} />
        ) : (
          <Form
            name="normal_login"
            className="login-form"
            initialValues={{
              remember: true,
            }}
            onFinish={this.handleSubmit}
            onFinishFailed={onFinishFailed}
          >
            <Form.Item
              name="username"
              rules={[
                {
                  required: true,
                  message: "Please input your Username!",
                },
              ]}
            >
              <Input
                prefix={<UserOutlined className="site-form-item-icon" />}
                placeholder="Username"
              />
            </Form.Item>
            <Form.Item
              name="password"
              rules={[
                {
                  required: true,
                  message: "Please input your Password!",
                },
              ]}
            >
              <Input
                prefix={<LockOutlined className="site-form-item-icon" />}
                type="password"
                placeholder="Password"
              />
            </Form.Item>
            <Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                style={{ marginRight: "10px" }}
              >
                Login
              </Button>
              Or
              <NavLink style={{ marginRight: "10px" }} to="/signup/">
                {" "}
                Signup
              </NavLink>
            </Form.Item>
          </Form>
        )}
      </div>
    );
  }
}


const WrappedNormalLoginForm = NormalLoginForm;


const mapStateToProps = (state) => {
  return {
    loading: state.loading,
    error: state.error,
  };
};

const mapDispatchToProps = dispatch => {

    return {
        onAuth: (username, password) => dispatch(actions.authLogin(username, password))
    }
} 

export default connect(mapStateToProps, mapDispatchToProps)(WrappedNormalLoginForm);